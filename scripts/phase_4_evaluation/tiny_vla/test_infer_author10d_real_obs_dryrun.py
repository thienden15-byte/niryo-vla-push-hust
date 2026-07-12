import os
import sys
import time
import pickle
import argparse
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


def find_usb_camera():
    candidates = []
    for i in range(20):
        name_path = Path(f"/sys/class/video4linux/video{i}/name")
        if name_path.exists():
            name = name_path.read_text(errors="ignore").strip()
            candidates.append((i, name))
            if "USB Camera2" in name or "USB" in name:
                return i, name, candidates
    if candidates:
        return candidates[0][0], candidates[0][1], candidates
    return None, None, candidates


def capture_frame(camera_index, out_dir):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_bgr = None
    for _ in range(15):
        ok, frame_bgr = cap.read()
        time.sleep(0.03)

    cap.release()

    if frame_bgr is None:
        raise RuntimeError("Camera opened but no frame captured")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"author10d_real_obs_{ts}.jpg"
    cv2.imwrite(str(raw_path), frame_bgr)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Theo fake test, model preprocess ra 336x336 sau expand2square.
    # Ở đây đưa ảnh thật về 180x320 giống thông báo train trước đó.
    frame_small = cv2.resize(frame_rgb, (320, 180), interpolation=cv2.INTER_AREA)

    img = torch.from_numpy(frame_small).permute(2, 0, 1).float() / 255.0

    # Dataset hiện có 1 camera front, runtime model cần image + image_r.
    # Ta duplicate front thành 2 view như lúc train/infer patch.
    curr_image = torch.stack([img, img], dim=0)  # [2,3,180,320]

    return curr_image, raw_path, frame_bgr.shape


def read_robot_joints(ip):
    from pyniryo import NiryoRobot

    robot = NiryoRobot(ip)
    joints = robot.get_joints()

    pose = None
    try:
        pose = robot.get_pose()
    except Exception as e:
        print("WARNING: get_pose failed:", repr(e))

    try:
        robot.close_connection()
    except Exception:
        pass

    joints = np.array(joints, dtype=np.float32)
    if joints.shape != (6,):
        raise RuntimeError(f"Expected 6 joints, got shape {joints.shape}: {joints}")

    return joints, pose


def expand2square(imgs_chw, background_color):
    b, c, h, w = imgs_chw.shape
    max_dim = max(h, w)
    imgs_np = imgs_chw.permute(0, 2, 3, 1).detach().cpu().numpy()

    expanded = np.full(
        (b, max_dim, max_dim, c),
        background_color,
        dtype=np.float32,
    )

    if h == w:
        expanded = imgs_np
    elif h > w:
        offset = (max_dim - w) // 2
        expanded[:, :h, offset:offset + w, :] = imgs_np
    else:
        offset = (max_dim - h) // 2
        expanded[:, offset:offset + h, :w, :] = imgs_np

    return torch.tensor(expanded, dtype=imgs_chw.dtype, device=imgs_chw.device)


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
    base = Path(
        os.environ.get(
            "TINYVLA_MODEL_BASE",
            str(Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"),
        )
    ).expanduser()
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== AUTHOR 10D REAL OBS DRY-RUN: NO ROBOT MOVEMENT =====")
    print("ckpt:", ckpt)
    print("base:", base)
    print("robot_ip:", args.ip)
    print("instruction:", args.instruction)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print()
    print("===== CAMERA READ =====")
    cam_idx = args.camera_index
    if cam_idx < 0:
        cam_idx, cam_name, candidates = find_usb_camera()
        print("camera candidates:", candidates)
        print("auto selected:", cam_idx, cam_name)
    else:
        print("manual selected camera:", cam_idx)

    if cam_idx is None:
        raise RuntimeError("No camera found")

    curr_image, raw_path, raw_shape = capture_frame(
        cam_idx,
        Path(
            os.environ.get(
                "TINYVLA_OUTPUT_DIR",
                str(
                    Path(__file__).resolve().parents[3]
                    / "outputs/tiny_vla/captures_author10d"
                ),
            )
        ).expanduser(),
    )

    print("captured raw shape:", raw_shape)
    print("saved frame:", raw_path)
    print("model curr_image:", tuple(curr_image.shape), curr_image.dtype, float(curr_image.min()), float(curr_image.max()))

    print()
    print("===== ROBOT READ ONLY =====")
    joints6, pose = read_robot_joints(args.ip)

    # qpos 7D = 6 joints + gripper.
    # Dataset hiện gripper gần 0, dry-run dùng gripper=0.
    qpos7 = np.concatenate([joints6, np.array([0.0], dtype=np.float32)], axis=0)

    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

    print("raw joints6:", np.round(joints6, 6))
    print("qpos7 with gripper=0:", np.round(qpos7, 6))
    print("qpos_norm:", np.round(qpos_norm, 6))
    print("qpos_norm abs max:", float(np.max(np.abs(qpos_norm))))
    print("current pose object:", pose)

    print()
    print("===== LOAD MODEL =====")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

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
    print("device:", model_device)
    print("dtype:", model_dtype)
    print("config state_dim:", getattr(model.config, "state_dim", None))
    print("config action_dim:", getattr(model.config, "action_dim", None))
    print("config chunk_size:", getattr(model.config, "chunk_size", None))

    print()
    print("===== PREPROCESS IMAGE =====")
    curr_image = curr_image.to(model_device, dtype=torch.float32)
    image, image_r = torch.chunk(curr_image, 2, dim=0)

    image = expand2square(image, tuple(x for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(
        image,
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"].to(model_device, dtype=model_dtype)

    image_r = expand2square(image_r, tuple(x for x in image_processor.image_mean))
    image_tensor_r = image_processor.preprocess(
        image_r,
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"].to(model_device, dtype=model_dtype)

    print("image_tensor:", tuple(image_tensor.shape), image_tensor.dtype)
    print("image_tensor_r:", tuple(image_tensor_r.shape), image_tensor_r.dtype)

    print()
    print("===== PROMPT =====")
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

    print("input_ids shape:", tuple(input_ids.shape))
    print("robot_state shape:", tuple(robot_state.shape))

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
    action_min = stats["action_min"].reshape(1, 1, -1)
    action_max = stats["action_max"].reshape(1, 1, -1)

    actions_real = actions_norm * action_std + action_mean
    within = np.logical_and(actions_real >= action_min, actions_real <= action_max)

    print()
    print("===== REAL OBS DRY-RUN RESULT: NO MOVEMENT =====")
    print("actions shape:", tuple(actions.shape))
    print("within dataset range ratio:", float(within.mean()))

    print()
    print("first action 10D:")
    print(np.round(actions_real[0, 0], 6))

    print()
    print("first 5 actions 10D:")
    print(np.round(actions_real[0, :5], 6))

    print()
    print("action real min per dim:")
    print(np.round(actions_real.min(axis=(0, 1)), 6))
    print("action real max per dim:")
    print(np.round(actions_real.max(axis=(0, 1)), 6))

    print()
    print("10D meaning:")
    print("[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]")

    print()
    print("IMPORTANT:")
    print("This script DID NOT move the robot.")
    print("Next step will inspect/convert 10D pose before any execution.")

    print()
    print("VRAM allocated MB:", round(torch.cuda.memory_allocated() / 1024 / 1024, 2))
    print("AUTHOR 10D REAL OBS DRY-RUN DONE")


if __name__ == "__main__":
    main()
