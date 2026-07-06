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


def decode_diffusion_minmax(raw, stats):
    action_min = stats["action_min"].reshape(1, 1, -1)
    action_max = stats["action_max"].reshape(1, 1, -1)
    return ((raw + 1.0) / 2.0) * (action_max - action_min) + action_min


def decode_old_meanstd(raw, stats):
    action_mean = stats["action_mean"].reshape(1, 1, -1)
    action_std = stats["action_std"].reshape(1, 1, -1)
    return raw * action_std + action_mean


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


def infer_once(frame_bgr, joints6, tokenizer, model, image_processor, input_ids, attention_mask, stats):
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    qpos7 = np.concatenate([joints6, np.array([0.0], dtype=np.float32)], axis=0)

    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

    robot_state = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(model_device, dtype=model_dtype)

    curr_image = base_live.frame_to_model_tensor(frame_bgr)
    image_tensor, image_tensor_r = base_live.preprocess_image(
        curr_image,
        image_processor,
        model_device,
        model_dtype,
    )

    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        images_r=image_tensor_r,
        states=robot_state,
    )

    with torch.no_grad():
        raw = model(**batch, eval=True)

    raw_np = raw.detach().float().cpu().numpy()
    return raw_np, qpos_norm


def temporal_aggregate_for_t(all_time_actions, t, k=0.01):
    """
    all_time_actions shape: [max_t, max_t + chunk_size, action_dim]
    At timestep t, lấy tất cả dự đoán trước đó cho thời điểm t.
    """
    actions_for_curr_step = all_time_actions[:, t]  # [max_t, action_dim]
    valid = np.any(np.abs(actions_for_curr_step) > 1e-9, axis=1)
    actions_populated = actions_for_curr_step[valid]

    if len(actions_populated) == 0:
        return None, 0

    # Giống ý tưởng tác giả: exp weights, ưu tiên dự đoán mới hơn.
    weights = np.exp(-k * np.arange(len(actions_populated)))
    weights = weights / weights.sum()
    weights = weights[:, None]

    return np.sum(actions_populated * weights, axis=0), len(actions_populated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")
    parser.add_argument("--dry-steps", type=int, default=12)
    parser.add_argument("--query-frequency", type=int, default=1)
    parser.add_argument("--temporal-k", type=float, default=0.01)
    args = parser.parse_args()

    ckpt = Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
    model_base = Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== OFFICIAL-STYLE DRY-RUN: NO MOVEMENT =====")
    print("instruction:", args.instruction)
    print("dry_steps:", args.dry_steps)
    print("query_frequency:", args.query_frequency)
    print("temporal_k:", args.temporal_k)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print()
    print("===== CAMERA CAPTURE =====")
    if args.camera_index < 0:
        cam_idx, cam_name, candidates = base_live.find_usb_camera()
        print("camera candidates:", candidates)
        print("auto selected:", cam_idx, cam_name)
    else:
        cam_idx = args.camera_index
        print("manual selected:", cam_idx)

    frame_bgr = capture_one_frame(cam_idx)

    out_dir = Path.home() / "tinyvla_niryo_runtime/official_style_dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / f"official_style_frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(frame_path), frame_bgr)
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

    print("current pose [x,y,z,r,p,y]:")
    print(np.round(current_pose, 6))

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

    print("model loaded OK")
    print("config action_head_type:", getattr(model.config, "action_head_type", None))
    print("config action_dim:", getattr(model.config, "action_dim", None))
    print("config chunk_size:", getattr(model.config, "chunk_size", None))

    model_device = next(model.parameters()).device

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

    chunk_size = int(getattr(model.config, "chunk_size", 16))
    action_dim = int(getattr(model.config, "action_dim", 10))

    max_t = args.dry_steps
    all_time_actions_raw = np.zeros((max_t, max_t + chunk_size, action_dim), dtype=np.float32)

    print()
    print("===== DRY ROLLOUT WITH TEMPORAL AGGREGATION =====")
    print("NOTE: Robot is NOT moving. Observation is fixed.")
    print()

    last_raw_chunk = None

    for t in range(max_t):
        if t % args.query_frequency == 0:
            raw_np, qpos_norm = infer_once(
                frame_bgr,
                joints6,
                tokenizer,
                model,
                image_processor,
                input_ids,
                attention_mask,
                stats,
            )

            raw_chunk = raw_np[0]  # [chunk_size, action_dim]
            last_raw_chunk = raw_chunk
            all_time_actions_raw[t, t:t + chunk_size, :] = raw_chunk

        agg_raw, n_votes = temporal_aggregate_for_t(all_time_actions_raw, t, k=args.temporal_k)

        if agg_raw is None:
            print(f"t={t:02d}: no action")
            continue

        # Decode action sau aggregation bằng min/max.
        agg_raw_reshaped = agg_raw.reshape(1, 1, -1)
        agg_minmax = decode_diffusion_minmax(agg_raw_reshaped, stats)[0, 0]
        agg_meanstd = decode_old_meanstd(agg_raw_reshaped, stats)[0, 0]

        cur_xyz = current_pose[:3]
        xyz_minmax = agg_minmax[:3]
        xyz_meanstd = agg_meanstd[:3]

        print(f"--- t={t:02d} votes={n_votes} ---")
        print("agg raw first3:", np.round(agg_raw[:3], 6))
        print("MINMAX xyz:", np.round(xyz_minmax, 6), "dxyz:", np.round(xyz_minmax - cur_xyz, 6))
        print("OLD meanstd xyz:", np.round(xyz_meanstd, 6), "dxyz:", np.round(xyz_meanstd - cur_xyz, 6))

    print()
    print("===== LAST RAW CHUNK DECODED BY MINMAX =====")
    if last_raw_chunk is not None:
        last_decoded = decode_diffusion_minmax(last_raw_chunk.reshape(1, chunk_size, action_dim), stats)[0]
        print("first 16 xyz minmax:")
        print(np.round(last_decoded[:, :3], 6))
        print("xyz min:", np.round(last_decoded[:, :3].min(axis=0), 6))
        print("xyz max:", np.round(last_decoded[:, :3].max(axis=0), 6))

    print()
    print("IMPORTANT:")
    print("This was dry-run only. Robot DID NOT move.")
    print("Next step: if minmax + temporal aggregation looks sane, create 1-step execution.")
    print("OFFICIAL-STYLE DRY-RUN DONE")


if __name__ == "__main__":
    main()
