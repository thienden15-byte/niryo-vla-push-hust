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

TINYVLA_REPO = Path(
    os.environ.get("TINYVLA_REPO", str(Path.home() / "TinyVLA"))
).expanduser()

sys.path.insert(0, str(TINYVLA_REPO))
sys.path.insert(0, str(TINYVLA_REPO / "llava-pythia"))

from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.conversation import conv_templates
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava_pythia.mm_utils import tokenizer_image_token

# official TinyVLA utility
from torch_utils import rot_6d_to_euler_angles

base_live_path = Path(__file__).resolve().parents[3] / "scripts/common/tiny_vla/run_author10d_fixed50_xyz_chunk_live.py"
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
        raise RuntimeError("Cannot capture camera frame")

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")
    args = parser.parse_args()

    ckpt = Path(
        os.environ.get(
            "TINYVLA_MODEL_PATH",
            str(Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"),
        )
    ).expanduser()
    model_base = Path(
        os.environ.get(
            "TINYVLA_MODEL_BASE",
            str(Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"),
        )
    ).expanduser()
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== INSPECT ROT6D INTERLEAVED CONVERSION: NO MOVEMENT =====")
    print("instruction:", args.instruction)

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
    current_rpy = current_pose[3:6]

    print("current xyz:", np.round(current_xyz, 6))
    print("current rpy:", np.round(current_rpy, 6))

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
    print("===== INFERENCE =====")
    with torch.no_grad():
        raw = model(**batch, eval=True)

    raw_np = raw.detach().float().cpu().numpy()
    decoded = decode_minmax(raw_np, stats)[0]  # [16,10]

    xyz = decoded[:, :3]
    rot6d = decoded[:, 3:9]
    grip = decoded[:, 9]

    rot6d_t = torch.from_numpy(rot6d).float()

    with torch.no_grad():
        rpy_t = rot_6d_to_euler_angles(rot6d_t, convention="XYZ")

    rpy = rpy_t.detach().cpu().numpy()

    print("===== FIRST 16 ACTIONS: XYZ + INTERLEAVED RPY =====")
    print("idx | xyz                         | rpy official                | delta_rpy_from_current")
    print("-" * 105)

    for i in range(decoded.shape[0]):
        drpy = rpy[i] - current_rpy
        print(
            f"{i:02d}  | "
            f"{np.round(xyz[i], 6)} | "
            f"{np.round(rpy[i], 6)} | "
            f"{np.round(drpy, 6)}"
        )

    print()
    print("===== SUMMARY =====")
    print("current rpy:", np.round(current_rpy, 6))
    print("rpy min:", np.round(rpy.min(axis=0), 6))
    print("rpy max:", np.round(rpy.max(axis=0), 6))
    print("abs delta rpy max:", np.round(np.max(np.abs(rpy - current_rpy), axis=0), 6))
    print("xyz min:", np.round(xyz.min(axis=0), 6))
    print("xyz max:", np.round(xyz.max(axis=0), 6))
    print("gripper min/max:", float(grip.min()), float(grip.max()))

    print()
    print("INTERPRETATION GUIDE:")
    print("- Nếu RPY lệch rất lớn, ví dụ roll/pitch/yaw nhảy > 1 rad, chưa chạy full 10D.")
    print("- Nếu RPY gần current pose, có thể thử official full action sau này.")
    print("- Script này KHÔNG di chuyển robot.")
    print("ROT6D INSPECT DONE")


if __name__ == "__main__":
    main()
