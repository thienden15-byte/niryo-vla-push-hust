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


def decode_minmax(raw, stats):
    action_min = stats["action_min"].reshape(1, 1, -1)
    action_max = stats["action_max"].reshape(1, 1, -1)
    return ((raw + 1.0) / 2.0) * (action_max - action_min) + action_min


def capture_one_frame(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(15):
        ok, frame = cap.read()
        time.sleep(0.03)

    cap.release()

    if frame is None:
        raise RuntimeError("Cannot capture frame")

    return frame


def clip_dxyz(dxyz, max_step):
    norm = float(np.linalg.norm(dxyz))
    if norm <= max_step:
        return dxyz, norm, False
    return dxyz / norm * max_step, norm, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-step", type=float, default=0.018)

    parser.add_argument("--min-z", type=float, default=0.055)
    parser.add_argument("--max-x", type=float, default=0.320)
    parser.add_argument("--max-abs-y", type=float, default=0.180)

    parser.add_argument("--max-joint-delta", type=float, default=0.30)
    parser.add_argument("--max-wrist-delta", type=float, default=0.25)
    args = parser.parse_args()

    ckpt = Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
    model_base = Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== MINMAX XYZ + IK GUARD PREVIEW: NO MOVEMENT =====")
    print("instruction:", args.instruction)
    print("samples:", args.samples)
    print("max_step:", args.max_step)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print()
    print("===== CAMERA =====")
    if args.camera_index < 0:
        cam_idx, cam_name, candidates = base_live.find_usb_camera()
        print("camera candidates:", candidates)
        print("auto selected:", cam_idx, cam_name)
    else:
        cam_idx = args.camera_index
        print("manual selected:", cam_idx)

    frame_bgr = capture_one_frame(cam_idx)

    out_dir = Path.home() / "tinyvla_niryo_runtime/minmax_xyz_ik_guard"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / f"preview_frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(frame_path), frame_bgr)
    print("saved frame:", frame_path)

    print()
    print("===== ROBOT READ + IK CHECK =====")
    from pyniryo import NiryoRobot

    robot = NiryoRobot(args.ip)
    try:
        joints6 = np.array(robot.get_joints(), dtype=np.float32)
        pose = robot.get_pose()
        current_pose = base_live.pose_to_list(pose)

        if current_pose is None:
            raise RuntimeError(f"Cannot parse current pose: {pose}")

        current_pose = np.array(current_pose, dtype=np.float64)
        current_xyz = current_pose[:3]
        current_rpy = current_pose[3:6]

        print("current joints:")
        print(np.round(joints6, 6))
        print("current pose:")
        print(np.round(current_pose, 6))

        ik_current = np.array(robot.inverse_kinematics(current_pose.tolist()), dtype=np.float64)
        print("IK(current pose) - current_joints:")
        print(np.round(ik_current - joints6, 6))
    finally:
        try:
            robot.close_connection()
        except Exception:
            pass

    qpos7 = np.concatenate([joints6, np.array([0.0], dtype=np.float32)], axis=0)
    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

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
    print("action_head_type:", getattr(model.config, "action_head_type", None))
    print("chunk_size:", getattr(model.config, "chunk_size", None))

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

    print()
    print("===== MODEL SAMPLES MINMAX =====")

    xyz_samples = []

    for i in range(args.samples):
        with torch.no_grad():
            raw = model(**batch, eval=True)

        raw_np = raw.detach().float().cpu().numpy()
        decoded = decode_minmax(raw_np, stats)[0]
        xyz0 = decoded[0, :3].astype(np.float64)

        xyz_samples.append(xyz0)

        print(
            f"sample {i}: xyz0={np.round(xyz0, 6)} "
            f"dxyz={np.round(xyz0 - current_xyz, 6)}"
        )

    xyz_samples = np.stack(xyz_samples, axis=0)

    # median chống outlier tốt hơn mean trong diffusion sampling
    raw_target_xyz = np.median(xyz_samples, axis=0)
    mean_xyz = np.mean(xyz_samples, axis=0)
    std_xyz = np.std(xyz_samples, axis=0)

    raw_dxyz = raw_target_xyz - current_xyz
    clipped_dxyz, raw_norm, was_clipped = clip_dxyz(raw_dxyz, args.max_step)
    target_xyz = current_xyz + clipped_dxyz

    target_pose = current_pose.copy()
    target_pose[:3] = target_xyz
    target_pose[3:6] = current_rpy

    print()
    print("===== SELECT TARGET =====")
    print("xyz mean  :", np.round(mean_xyz, 6))
    print("xyz median:", np.round(raw_target_xyz, 6))
    print("xyz std   :", np.round(std_xyz, 6))
    print("raw dxyz  :", np.round(raw_dxyz, 6), "norm:", round(raw_norm, 6))
    print("clipped?  :", was_clipped)
    print("final dxyz:", np.round(clipped_dxyz, 6), "norm:", round(float(np.linalg.norm(clipped_dxyz)), 6))
    print("target pose keep RPY:")
    print(np.round(target_pose, 6))

    print()
    print("===== ROUGH XYZ SAFETY =====")
    x, y, z = target_xyz
    xyz_ok = True

    if z < args.min_z:
        print("BAD: z below min_z")
        xyz_ok = False
    if x < 0.05 or x > args.max_x:
        print("BAD: x outside safe range")
        xyz_ok = False
    if abs(y) > args.max_abs_y:
        print("BAD: |y| outside safe range")
        xyz_ok = False

    if xyz_ok:
        print("XYZ safety OK")

    print()
    print("===== IK PREVIEW FOR TARGET =====")
    robot = NiryoRobot(args.ip)
    try:
        ik_target = np.array(robot.inverse_kinematics(target_pose.tolist()), dtype=np.float64)
    finally:
        try:
            robot.close_connection()
        except Exception:
            pass

    delta_j = ik_target - joints6
    print("ik target joints:")
    print(np.round(ik_target, 6))
    print("delta joints:")
    print(np.round(delta_j, 6))
    print("max abs delta all joints:", round(float(np.max(np.abs(delta_j))), 6))
    print("wrist delta j4/j5/j6:", np.round(delta_j[3:6], 6))

    joint_ok = True
    if np.max(np.abs(delta_j)) > args.max_joint_delta:
        print("BAD: some joint delta too large")
        joint_ok = False
    if np.max(np.abs(delta_j[3:6])) > args.max_wrist_delta:
        print("BAD: wrist joint delta too large")
        joint_ok = False

    if joint_ok:
        print("IK joint safety OK")

    print()
    print("===== DECISION =====")
    if xyz_ok and joint_ok:
        print("SAFE_TO_EXECUTE_ONE_SMALL_STEP = YES")
        print("Next step can execute by move_joints(ik_target), not move_pose.")
    else:
        print("SAFE_TO_EXECUTE_ONE_SMALL_STEP = NO")
        print("Do not execute. Need smaller max-step or different target selection.")

    print()
    print("IMPORTANT: This script DID NOT move the robot.")
    print("STEP 24 PREVIEW DONE")


if __name__ == "__main__":
    main()
