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

base_live_path = Path(__file__).resolve().parents[3] / "scripts/common/tiny_vla/run_author10d_fixed50_xyz_chunk_live.py"
spec = importlib.util.spec_from_file_location("base_live", base_live_path)
base_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_live)


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
    for _ in range(20):
        ok, frame = cap.read()
        time.sleep(0.03)

    cap.release()

    if frame is None:
        raise RuntimeError("Cannot capture camera frame")

    return frame


def detect_green_object(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Dải xanh lá tương đối rộng, dùng để debug/mask thôi.
    lower = np.array([35, 45, 45], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    area = float(cv2.contourArea(c))

    if area < 200:
        return None, mask, None

    x, y, w, h = cv2.boundingRect(c)

    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-6:
        cx, cy = x + w / 2, y + h / 2
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    box = (x, y, w, h, area, cx, cy)
    return c, mask, box


def make_variants(frame_bgr, box):
    h, w = frame_bgr.shape[:2]

    variants = {}

    variants["original"] = frame_bgr.copy()

    # Mask đúng vật xanh
    mask_green = frame_bgr.copy()
    if box is not None:
        x, y, bw, bh, area, cx, cy = box
        pad = 30
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        mask_green[y1:y2, x1:x2] = (128, 128, 128)
    variants["mask_green_object"] = mask_green

    # Mask vùng nền không liên quan, chọn góc trên phải.
    mask_background = frame_bgr.copy()
    bx1 = int(w * 0.70)
    by1 = int(h * 0.10)
    bx2 = int(w * 0.92)
    by2 = int(h * 0.32)
    mask_background[by1:by2, bx1:bx2] = (128, 128, 128)
    variants["mask_background"] = mask_background

    return variants


def draw_debug(frame_bgr, box, out_path):
    show = frame_bgr.copy()

    if box is not None:
        x, y, w, h, area, cx, cy = box
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(show, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        cv2.putText(
            show,
            f"green object cx={cx:.1f} cy={cy:.1f} area={area:.0f}",
            (x, max(30, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            show,
            "NO GREEN OBJECT DETECTED BY HSV DEBUG",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), show)


def infer_one(frame_bgr, tokenizer, model, image_processor, input_ids, attention_mask, robot_state, action_mean, action_std):
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

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
        actions = model(**batch, eval=True)

    actions_real = actions.detach().float().cpu().numpy() * action_std + action_mean
    return actions_real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")
    args = parser.parse_args()

    ckpt = Path(
        os.environ.get(
            "TINYVLA_MODEL_PATH",
            str(
                Path.home()
                / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
            ),
        )
    ).expanduser()
    model_base = Path(
        os.environ.get(
            "TINYVLA_MODEL_BASE",
            str(
                Path.home()
                / "TinyVLA/pretrained/Llava-Pythia-400M"
            ),
        )
    ).expanduser()
    stats_path = ckpt / "dataset_stats.pkl"
    out_dir = Path(
        os.environ.get(
            "TINYVLA_OUTPUT_DIR",
            str(
                Path(__file__).resolve().parents[3]
                / "outputs/tiny_vla/object_sensitivity_debug"
            ),
        )
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== OBJECT SENSITIVITY TEST: NO MOVEMENT =====")
    print("instruction:", args.instruction)

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

    ts = time.strftime("%Y%m%d_%H%M%S")
    original_path = out_dir / f"original_{ts}.jpg"
    cv2.imwrite(str(original_path), frame_bgr)

    contour, mask, box = detect_green_object(frame_bgr)
    debug_path = out_dir / f"green_detect_debug_{ts}.jpg"
    draw_debug(frame_bgr, box, debug_path)

    print("saved original:", original_path)
    print("saved green debug:", debug_path)

    if box is None:
        print("HSV debug could not detect green object.")
    else:
        x, y, w, h, area, cx, cy = box
        print("HSV green box x,y,w,h,area,cx,cy:")
        print(np.round([x, y, w, h, area, cx, cy], 2))

    variants = make_variants(frame_bgr, box)

    for name, img in variants.items():
        p = out_dir / f"{name}_{ts}.jpg"
        cv2.imwrite(str(p), img)
        print("saved variant:", name, p)

    print()
    print("===== ROBOT STATE READ ONLY =====")
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
    current_pose = np.array(current_pose, dtype=np.float64)

    print("current pose:")
    print(np.round(current_pose, 6))

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

    action_mean = stats["action_mean"].reshape(1, 1, -1)
    action_std = stats["action_std"].reshape(1, 1, -1)

    print("model loaded OK")

    print()
    print("===== RUN INFERENCE ON IMAGE VARIANTS =====")

    results = {}
    for name, img in variants.items():
        actions_real = infer_one(
            img,
            tokenizer,
            model,
            image_processor,
            input_ids,
            attention_mask,
            robot_state,
            action_mean,
            action_std,
        )
        results[name] = actions_real
        print()
        print(f"--- {name} ---")
        print("action0 xyz:", np.round(actions_real[0, 0, :3], 6))
        print("action0 10D:", np.round(actions_real[0, 0], 6))
        print("first 5 xyz:")
        print(np.round(actions_real[0, :5, :3], 6))

    print()
    print("===== SENSITIVITY SUMMARY =====")
    orig = results["original"][0, 0, :3]

    for name in ["mask_green_object", "mask_background"]:
        xyz = results[name][0, 0, :3]
        diff = xyz - orig
        print()
        print(f"{name} - original action0 xyz diff:")
        print(np.round(diff, 6), "norm:", round(float(np.linalg.norm(diff)), 6))

    print()
    print("INTERPRETATION:")
    print("- Nếu mask_green_object làm action đổi mạnh hơn mask_background, model có khả năng đang dùng thông tin vật xanh.")
    print("- Nếu mask_green_object gần như không đổi, model có thể chưa thật sự bám vật xanh.")
    print("- HSV box chỉ là debug để biết vật xanh nằm ở đâu trong ảnh; nó KHÔNG điều khiển robot.")
    print()
    print("OBJECT SENSITIVITY TEST DONE")


if __name__ == "__main__":
    main()
