import os
import sys
import time
import pickle
import argparse
import threading
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


class LiveCamera:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.stop_flag = False
        self.user_stop = False
        self.status_lines = ["Starting live camera..."]
        self.window_name = "AUTHOR10D LIVE RUN - press q to stop"
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def update_status(self, lines):
        with self.lock:
            self.status_lines = list(lines)

    def get_frame(self, wait=True, timeout=5.0):
        t0 = time.time()
        while True:
            with self.lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()

            if not wait:
                return None

            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for camera frame")

            time.sleep(0.02)

    def should_stop(self):
        return self.user_stop or self.stop_flag

    def close(self):
        self.stop_flag = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        frame_id = 0

        while not self.stop_flag:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            frame_id += 1

            with self.lock:
                self.latest_frame = frame.copy()
                lines = list(self.status_lines)

            show = frame.copy()

            overlay = show.copy()
            cv2.rectangle(overlay, (10, 10), (1260, 170), (0, 0, 0), -1)
            show = cv2.addWeighted(overlay, 0.35, show, 0.65, 0)

            cv2.putText(
                show,
                f"AUTHOR10D LIVE | frame={frame_id} | q/ESC stop",
                (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            y = 78
            for line in lines[:4]:
                cv2.putText(
                    show,
                    str(line),
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 30

            cv2.imshow(self.window_name, show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                self.user_stop = True
                self.stop_flag = True
                break

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def frame_to_model_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_small = cv2.resize(frame_rgb, (320, 180), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(frame_small).permute(2, 0, 1).float() / 255.0

    # 1 camera front, duplicate thành image + image_r
    curr_image = torch.stack([img, img], dim=0)
    return curr_image


def expand2square(imgs_chw, background_color):
    b, c, h, w = imgs_chw.shape
    max_dim = max(h, w)
    imgs_np = imgs_chw.permute(0, 2, 3, 1).detach().cpu().numpy()

    expanded = np.full((b, max_dim, max_dim, c), background_color, dtype=np.float32)

    if h == w:
        expanded = imgs_np
    elif h > w:
        offset = (max_dim - w) // 2
        expanded[:, :h, offset:offset + w, :] = imgs_np
    else:
        offset = (max_dim - h) // 2
        expanded[:, offset:offset + h, :w, :] = imgs_np

    return torch.tensor(expanded, dtype=imgs_chw.dtype, device=imgs_chw.device)


def preprocess_image(curr_image, image_processor, model_device, model_dtype):
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

    return image_tensor, image_tensor_r


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


def clip_dxyz(dxyz, max_norm=0.012, max_abs=0.010, max_down=0.005):
    d = np.array(dxyz, dtype=np.float64)

    d = np.clip(d, -max_abs, max_abs)

    if d[2] < -max_down:
        d[2] = -max_down

    norm = float(np.linalg.norm(d))
    if norm > max_norm:
        d = d / norm * max_norm

    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="169.254.200.200")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--instruction", default="push the green object to the right")

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--actions-per-infer", type=int, default=4)

    parser.add_argument("--max-step-norm", type=float, default=0.012)
    parser.add_argument("--max-step-axis", type=float, default=0.010)
    parser.add_argument("--max-down", type=float, default=0.005)

    parser.add_argument("--velocity", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.20)

    parser.add_argument("--min-z", type=float, default=0.120)
    parser.add_argument("--max-x", type=float, default=0.300)
    parser.add_argument("--max-abs-y", type=float, default=0.160)

    parser.add_argument("--no-back-x", action="store_true")

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if not args.execute or args.confirm != "YES_RUN_FIXED50_LIVE":
        raise RuntimeError("Refuse to run. Need --execute --confirm YES_RUN_FIXED50_LIVE")

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
    out_dir = Path(
        os.environ.get(
            "TINYVLA_OUTPUT_DIR",
            str(
                Path(__file__).resolve().parents[3]
                / "outputs/tiny_vla/captures_author10d_live"
            ),
        )
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== AUTHOR10D FIXED50 XYZ-ONLY WITH LIVE CAMERA =====")
    print("steps:", args.steps)
    print("actions_per_infer:", args.actions_per_infer)
    print("robot_ip:", args.ip)
    print("instruction:", args.instruction)
    print("max_step_norm:", args.max_step_norm)
    print("max_step_axis:", args.max_step_axis)
    print("max_down:", args.max_down)
    print("min_z:", args.min_z)
    print("max_x:", args.max_x)
    print("max_abs_y:", args.max_abs_y)
    print("no_back_x:", args.no_back_x)
    print()

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print("===== CAMERA OPEN =====")
    cam_idx = args.camera_index

    if cam_idx < 0:
        cam_idx, cam_name, candidates = find_usb_camera()
        print("camera candidates:", candidates)
        print("auto selected:", cam_idx, cam_name)
    else:
        print("manual selected:", cam_idx)

    if cam_idx is None:
        raise RuntimeError("No camera found")

    live = LiveCamera(cam_idx)
    live.start()
    live.update_status(["Loading robot/model...", "Robot not moving yet"])

    print()
    print("===== ROBOT CONNECT =====")
    from pyniryo import NiryoRobot

    robot = NiryoRobot(args.ip)
    robot.set_learning_mode(False)
    robot.set_arm_max_velocity(args.velocity)

    print()
    print("===== LOAD MODEL ONCE =====")
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

    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)

    action_mean = stats["action_mean"].reshape(1, 1, -1)
    action_std = stats["action_std"].reshape(1, 1, -1)

    global_step = 0
    cycle = 0

    try:
        while global_step < args.steps:
            if live.should_stop():
                print("STOP requested from live camera window.")
                break

            cycle += 1
            print()
            print(f"===== INFER CYCLE {cycle} | global_step {global_step}/{args.steps} =====")

            live.update_status([
                f"infer cycle={cycle} step={global_step}/{args.steps}",
                "Capturing current camera frame...",
                "Robot will use XYZ-only, keep RPY",
            ])

            frame_bgr = live.get_frame(wait=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            frame_path = out_dir / f"live_cycle{cycle:03d}_step{global_step:03d}_{ts}.jpg"
            cv2.imwrite(str(frame_path), frame_bgr)

            curr_image = frame_to_model_tensor(frame_bgr)
            image_tensor, image_tensor_r = preprocess_image(
                curr_image,
                image_processor,
                model_device,
                model_dtype,
            )

            joints6 = np.array(robot.get_joints(), dtype=np.float32)
            pose = robot.get_pose()
            current_pose = pose_to_list(pose)

            if current_pose is None:
                raise RuntimeError(f"Cannot parse current pose: {pose}")

            current_pose = np.array(current_pose, dtype=np.float64)

            qpos7 = np.concatenate([joints6, np.array([0.0], dtype=np.float32)], axis=0)
            qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

            robot_state = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(
                model_device,
                dtype=model_dtype,
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

            actions_norm = actions.detach().float().cpu().numpy()
            actions_real = actions_norm * action_std + action_mean

            actions_to_run = min(
                args.actions_per_infer,
                actions_real.shape[1],
                args.steps - global_step,
            )

            print("saved frame:", frame_path)
            print("chunk actions shape:", actions_real.shape)
            print("actions_to_run this cycle:", actions_to_run)

            for local_i in range(actions_to_run):
                if live.should_stop():
                    print("STOP requested from live camera window.")
                    global_step = args.steps
                    break

                global_step += 1
                print()
                print(f"----- EXEC FIXED STEP {global_step}/{args.steps} | chunk action {local_i+1}/{actions_to_run} -----")

                pose = robot.get_pose()
                current_pose = pose_to_list(pose)

                if current_pose is None:
                    raise RuntimeError(f"Cannot parse current pose during chunk: {pose}")

                current_pose = np.array(current_pose, dtype=np.float64)
                current_xyz = current_pose[:3]

                action_i = actions_real[0, local_i]
                target_xyz_model = action_i[:3].astype(np.float64)

                raw_dxyz = target_xyz_model - current_xyz

                if args.no_back_x and raw_dxyz[0] < 0:
                    raw_dxyz[0] = 0.0

                safe_dxyz = clip_dxyz(
                    raw_dxyz,
                    max_norm=args.max_step_norm,
                    max_abs=args.max_step_axis,
                    max_down=args.max_down,
                )

                target_pose_safe = current_pose.copy()
                target_pose_safe[:3] = current_xyz + safe_dxyz
                target_pose_safe[3:6] = current_pose[3:6]

                live.update_status([
                    f"RUN step={global_step}/{args.steps} cycle={cycle}",
                    f"cur xyz={np.round(current_xyz, 3)}",
                    f"model xyz={np.round(target_xyz_model, 3)}",
                    f"safe dxyz={np.round(safe_dxyz, 3)}",
                ])

                print("current xyz:", np.round(current_xyz, 6))
                print("model target xyz:", np.round(target_xyz_model, 6))
                print("raw dxyz:", np.round(raw_dxyz, 6), "norm:", round(float(np.linalg.norm(raw_dxyz)), 6))
                print("safe dxyz:", np.round(safe_dxyz, 6), "norm:", round(float(np.linalg.norm(safe_dxyz)), 6))
                print("target pose safe:", np.round(target_pose_safe, 6))

                if target_pose_safe[2] < args.min_z:
                    print("STOP: target z below min_z")
                    global_step = args.steps
                    break

                if target_pose_safe[0] > args.max_x:
                    print("STOP: target x above max_x")
                    global_step = args.steps
                    break

                if abs(target_pose_safe[1]) > args.max_abs_y:
                    print("STOP: target |y| above max_abs_y")
                    global_step = args.steps
                    break

                robot.move_pose(*target_pose_safe.tolist())
                time.sleep(args.sleep)

                new_pose = pose_to_list(robot.get_pose())
                print("new pose:", None if new_pose is None else np.round(new_pose, 6))

        print()
        print("AUTHOR10D FIXED50 XYZ-ONLY WITH LIVE CAMERA DONE")

    finally:
        live.update_status(["Stopping...", "Closing robot/camera"])
        try:
            robot.close_connection()
        except Exception:
            pass

        try:
            live.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
