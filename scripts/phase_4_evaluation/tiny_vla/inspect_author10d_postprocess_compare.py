import os
import sys
import time
import pickle
import argparse
import importlib.util
from pathlib import Path

import cv2
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


base_live_path = Path.home() / "tinyvla_niryo_runtime/scripts/run_author10d_fixed50_xyz_chunk_live.py"
spec = importlib.util.spec_from_file_location("base_live", base_live_path)
base_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_live)


def capture_one_frame(camera_index, out_dir):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(20):
        ok, frame = cap.read()
        time.sleep(0.03)

    cap.release()

    if frame is None:
        raise RuntimeError("Cannot capture frame")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = out_dir / f"postprocess_compare_{ts}.jpg"
    cv2.imwrite(str(p), frame)
    return frame, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")
    args = parser.parse_args()

    ckpt = Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
    model_base = Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== AUTHOR10D POSTPROCESS COMPARE: NO MOVEMENT =====")
    print("instruction:", args.instruction)
    print("ckpt:", ckpt)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print()
    print("===== STATS KEYS / SHAPES =====")
    for k in ["action_mean", "action_std", "action_min", "action_max", "qpos_mean", "qpos_std"]:
        v = stats[k]
        print(k, getattr(v, "shape", None), np.round(v, 6))

    print()
    print("===== CAMERA =====")
    if args.camera_index < 0:
        cam_idx, cam_name, candidates = base_live.find_usb_camera()
        print("camera candidates:", candidates)
        print("auto selected:", cam_idx, cam_name)
    else:
        cam_idx = args.camera_index
        print("manual selected:", cam_idx)

    frame_bgr, frame_path = capture_one_frame(
        cam_idx,
        Path.home() / "tinyvla_niryo_runtime/postprocess_compare",
    )
    print("saved frame:", frame_path)

    print()
    print("===== ROBOT READ ONLY =====")
    from pyniryo import NiryoRobot

    robot = NiryoRobot(args.ip)
    try:
        joints6 = np.array(robot.get_joints(), dtype=np.float32)
        pose = robot.get_pose()
    finally:
        try:
            robot.close_connection()
        except Exception:
            pass

    current_pose = base_live.pose_to_list(pose)
    if current_pose is None:
        raise RuntimeError(f"Cannot parse current pose: {pose}")

    current_pose = np.array(current_pose, dtype=np.float64)
    current_xyz = current_pose[:3]

    print("current pose [x,y,z,r,p,y]:")
    print(np.round(current_pose, 6))

    qpos7 = np.concatenate([joints6, np.array([0.0], dtype=np.float32)], axis=0)

    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

    print("qpos7:", np.round(qpos7, 6))
    print("qpos_norm:", np.round(qpos_norm, 6))
    print("qpos_norm abs max:", float(np.max(np.abs(qpos_norm))))

    print()
    print("===== LOAD MODEL =====")
    torch.cuda.empty_cache()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=str(ckpt),
        model_base=str(model_base),
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
    print("device:", model_device)
    print("dtype:", model_dtype)
    print("config state_dim:", getattr(model.config, "state_dim", None))
    print("config action_dim:", getattr(model.config, "action_dim", None))
    print("config chunk_size:", getattr(model.config, "chunk_size", None))
    print("config action_head_type:", getattr(model.config, "action_head_type", None))

    print()
    print("===== PREPROCESS IMAGE / TEXT / STATE =====")

    curr_image = base_live.frame_to_model_tensor(frame_bgr)
    image_tensor, image_tensor_r = base_live.preprocess_image(
        curr_image,
        image_processor,
        model_device,
        model_dtype,
    )

    conv = conv_templates["pythia"].copy()
    inp = args.instruction

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt() + " <|endoftext|>"

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

    print("input_ids:", tuple(input_ids.shape))
    print("image_tensor:", tuple(image_tensor.shape), image_tensor.dtype)
    print("robot_state:", tuple(robot_state.shape), robot_state.dtype)

    print()
    print("===== RUN MODEL INFERENCE =====")
    with torch.no_grad():
        actions_raw = model(**batch, eval=True)

    raw = actions_raw.detach().float().cpu().numpy()

    print("raw actions shape:", raw.shape)
    print("raw min/max per dim:")
    print("min:", np.round(raw.min(axis=(0, 1)), 6))
    print("max:", np.round(raw.max(axis=(0, 1)), 6))
    print("raw action0:")
    print(np.round(raw[0, 0], 6))

    action_mean = stats["action_mean"].reshape(1, 1, -1)
    action_std = stats["action_std"].reshape(1, 1, -1)
    action_min = stats["action_min"].reshape(1, 1, -1)
    action_max = stats["action_max"].reshape(1, 1, -1)

    decoded_meanstd = raw * action_std + action_mean
    decoded_minmax = ((raw + 1.0) / 2.0) * (action_max - action_min) + action_min

    print()
    print("===== COMPARE ACTION0 DECODE =====")
    print("current xyz:")
    print(np.round(current_xyz, 6))

    for name, arr in [
        ("MEAN_STD_DECODE_OLD", decoded_meanstd),
        ("MIN_MAX_DECODE_DIFFUSION_STYLE", decoded_minmax),
    ]:
        xyz0 = arr[0, 0, :3].astype(np.float64)
        dxyz0 = xyz0 - current_xyz

        print()
        print(f"--- {name} ---")
        print("action0 xyz:")
        print(np.round(xyz0, 6))
        print("dxyz from current:")
        print(np.round(dxyz0, 6), "norm:", round(float(np.linalg.norm(dxyz0)), 6))
        print("action0 10D:")
        print(np.round(arr[0, 0], 6))
        print("first 5 xyz:")
        print(np.round(arr[0, :5, :3], 6))
        print("xyz min over chunk:")
        print(np.round(arr[0, :, :3].min(axis=0), 6))
        print("xyz max over chunk:")
        print(np.round(arr[0, :, :3].max(axis=0), 6))

        warns = []
        x, y, z = xyz0
        if x < 0.05 or x > 0.32:
            warns.append("x outside rough safe range")
        if abs(y) > 0.18:
            warns.append("|y| outside rough safe range")
        if z < 0.055:
            warns.append("z below 0.055")
        if z > 0.25:
            warns.append("z too high")
        if warns:
            print("WARN:", "; ".join(warns))
        else:
            print("rough safety: OK")

    print()
    print("===== DIFFERENCE BETWEEN DECODES =====")
    diff = decoded_minmax - decoded_meanstd
    print("action0 xyz minmax - meanstd:")
    print(np.round(diff[0, 0, :3], 6), "norm:", round(float(np.linalg.norm(diff[0, 0, :3])), 6))
    print("mean abs diff all dims:", round(float(np.mean(np.abs(diff))), 6))
    print("max abs diff all dims:", round(float(np.max(np.abs(diff))), 6))

    print()
    print("IMPORTANT:")
    print("This script DID NOT move the robot.")
    print("If MIN_MAX looks more plausible, next runtime must use min/max postprocess.")
    print("POSTPROCESS COMPARE DONE")


if __name__ == "__main__":
    main()
