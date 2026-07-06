import os
import sys
import math
import pickle
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/TinyVLA"))
sys.path.insert(0, os.path.expanduser("~/TinyVLA/llava-pythia"))

from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.conversation import conv_templates
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava_pythia.mm_utils import tokenizer_image_token


# Import lại các hàm camera/robot/preprocess đã tạo ở Bước 6
dryrun_path = Path.home() / "tinyvla_niryo_runtime/scripts/test_infer_author10d_real_obs_dryrun.py"
spec = importlib.util.spec_from_file_location("author10d_dryrun", dryrun_path)
dry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dry)


def normalize(v, eps=1e-8):
    return v / max(float(np.linalg.norm(v)), eps)


def rot6d_to_matrix(rot6d):
    """
    rot6d: 6 values.
    Cách hiểu phổ biến: 6D là 2 cột đầu của rotation matrix.
    """
    a1 = np.asarray(rot6d[:3], dtype=np.float64)
    a2 = np.asarray(rot6d[3:6], dtype=np.float64)

    b1 = normalize(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)
    return R


def matrix_to_rpy(R):
    """
    Convert rotation matrix -> roll, pitch, yaw.
    Convention dùng phổ biến cho robot: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)


def pose_to_list(pose):
    if pose is None:
        return None

    if hasattr(pose, "to_list"):
        try:
            arr = pose.to_list()
            if len(arr) >= 6:
                return [float(x) for x in arr[:6]]
        except Exception:
            pass

    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    vals = []
    ok = True
    for n in names:
        if hasattr(pose, n):
            vals.append(float(getattr(pose, n)))
        else:
            ok = False
            break
    if ok:
        return vals

    if isinstance(pose, (list, tuple)) and len(pose) >= 6:
        return [float(x) for x in pose[:6]]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")
    args = parser.parse_args()

    ckpt = Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
    base = Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== AUTHOR 10D POSE CONVERSION INSPECT: NO MOVEMENT =====")
    print("robot_ip:", args.ip)
    print("instruction:", args.instruction)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print()
    print("===== CAMERA READ =====")
    cam_idx = args.camera_index
    if cam_idx < 0:
        cam_idx, cam_name, candidates = dry.find_usb_camera()
        print("camera candidates:", candidates)
        print("auto selected:", cam_idx, cam_name)
    else:
        print("manual selected:", cam_idx)

    curr_image, raw_path, raw_shape = dry.capture_frame(
        cam_idx,
        Path.home() / "tinyvla_niryo_runtime/captures_author10d",
    )

    print("captured raw shape:", raw_shape)
    print("saved frame:", raw_path)

    print()
    print("===== ROBOT READ ONLY =====")
    joints6, pose = dry.read_robot_joints(args.ip)
    current_pose = pose_to_list(pose)

    print("raw joints6:", np.round(joints6, 6))
    print("current pose object:", pose)
    print("current pose list [x,y,z,roll,pitch,yaw]:", None if current_pose is None else np.round(current_pose, 6))

    qpos7 = np.concatenate([joints6, np.array([0.0], dtype=np.float32)], axis=0)
    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

    print("qpos7:", np.round(qpos7, 6))
    print("qpos_norm abs max:", float(np.max(np.abs(qpos_norm))))

    print()
    print("===== LOAD MODEL =====")
    torch.cuda.empty_cache()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=str(ckpt),
        model_base=str(base),
        model_name="llava-pythia-lora",
        device="cuda",
        device_map="cuda",
    )

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    print("model loaded OK")
    print("config state_dim:", getattr(model.config, "state_dim", None))
    print("config action_dim:", getattr(model.config, "action_dim", None))
    print("config chunk_size:", getattr(model.config, "chunk_size", None))

    print()
    print("===== PREPROCESS IMAGE =====")
    curr_image = curr_image.to(model_device, dtype=torch.float32)
    image, image_r = torch.chunk(curr_image, 2, dim=0)

    image = dry.expand2square(image, tuple(x for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(
        image,
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"].to(model_device, dtype=model_dtype)

    image_r = dry.expand2square(image_r, tuple(x for x in image_processor.image_mean))
    image_tensor_r = image_processor.preprocess(
        image_r,
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"].to(model_device, dtype=model_dtype)

    conv = conv_templates["pythia"].copy()
    inp = args.instruction
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    prompt += " <|endoftext|>"

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model_device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    robot_state = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(model_device, dtype=model_dtype)

    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        images_r=image_tensor_r,
        states=robot_state,
    )

    print()
    print("===== RUN MODEL INFERENCE =====")
    with torch.no_grad():
        actions = model(**batch, eval=True)

    actions_norm = actions.detach().float().cpu().numpy()
    action_mean = stats["action_mean"].reshape(1, 1, -1)
    action_std = stats["action_std"].reshape(1, 1, -1)
    actions_real = actions_norm * action_std + action_mean

    print()
    print("===== CONVERT FIRST 5 ACTIONS 10D -> ROBOT POSE =====")
    print("10D = [x,y,z, rot6d_0..5, gripper]")
    print()

    converted = []

    for i in range(min(5, actions_real.shape[1])):
        a = actions_real[0, i]
        xyz = a[:3].astype(np.float64)
        rot6d = a[3:9].astype(np.float64)
        gripper = float(a[9])

        R = rot6d_to_matrix(rot6d)
        rpy = matrix_to_rpy(R)
        pose6 = np.concatenate([xyz, rpy], axis=0)
        converted.append(pose6)

        print(f"--- action {i+1} ---")
        print("raw action10:", np.round(a, 6))
        print("target xyz:", np.round(xyz, 6))
        print("target rpy rad:", np.round(rpy, 6))
        print("target rpy deg:", np.round(np.degrees(rpy), 2))
        print("gripper:", round(gripper, 6))

        if current_pose is not None:
            cur = np.array(current_pose[:6], dtype=np.float64)
            dxyz = xyz - cur[:3]
            drpy = rpy - cur[3:6]
            print("delta xyz from current:", np.round(dxyz, 6), "norm:", round(float(np.linalg.norm(dxyz)), 6))
            print("delta rpy from current rad:", np.round(drpy, 6))

        warn = []
        if xyz[0] < 0.05 or xyz[0] > 0.35:
            warn.append("x outside rough safe range [0.05, 0.35]")
        if xyz[1] < -0.25 or xyz[1] > 0.25:
            warn.append("y outside rough safe range [-0.25, 0.25]")
        if xyz[2] < 0.055 or xyz[2] > 0.25:
            warn.append("z outside rough safe range [0.055, 0.25]")
        if warn:
            print("WARN:", "; ".join(warn))
        else:
            print("rough pose range: OK")

        print()

    converted = np.array(converted)

    print("===== SUMMARY =====")
    print("converted pose shape:", converted.shape)
    print("xyz min:", np.round(converted[:, :3].min(axis=0), 6))
    print("xyz max:", np.round(converted[:, :3].max(axis=0), 6))

    print()
    print("IMPORTANT:")
    print("This script DID NOT move the robot.")
    print("If target pose and delta look reasonable, next step is 1-step safe execution with explicit confirm.")

    print()
    print("AUTHOR 10D POSE CONVERSION INSPECT DONE")


if __name__ == "__main__":
    main()
