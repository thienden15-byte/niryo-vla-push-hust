#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from pyniryo import NiryoRobot


# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Niryo miniVLA V5 Robot-Frame Grid Collector"
    )

    parser.add_argument("--robot-ip", default="169.254.200.200")
    parser.add_argument("--dataset-dir", default="dataset_push_real_v5")

    parser.add_argument("--instruction", default="push the object")

    # V5: vị trí vật theo robot-frame, KHÔNG phải camera-frame.
    parser.add_argument(
        "--robot-zone",
        default="unknown",
        choices=[
            "unknown",
            "near_left", "near_center", "near_right",
            "mid_left", "mid_center", "mid_right",
            "far_left", "far_center", "far_right",
        ],
        help="Vị trí vật theo robot-frame: near/mid/far + left/center/right",
    )

    parser.add_argument("--camera-index", type=int, default=3)
    parser.add_argument("--camera-name-keyword", default="USB Camera2")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-raw-jpg", action="store_true", default=True)
    parser.add_argument("--raw-jpg-width", type=int, default=640)

    parser.add_argument("--teach-hz", type=float, default=10.0)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--min-total-diff", type=float, default=0.02)

    parser.add_argument("--velocity", type=int, default=15)

    # Đây là số waypoint robot sẽ đi khi step-replay.
    # Không phải padding. Nếu teach ít hoặc nhiều điểm, quỹ đạo sẽ được chia lại thành số waypoint này.
    parser.add_argument("--replay-waypoints", type=int, default=100)

    parser.add_argument("--pause-after-move", type=float, default=0.05)
    parser.add_argument("--move-start-delay", type=float, default=0.5)

    parser.add_argument("--flip-camera-horizontal", action="store_true")
    parser.add_argument("--flip-camera-vertical", action="store_true")

    # HSV chỉ dùng để debug vị trí vật xanh, không dùng làm input chính cho miniVLA.
    parser.add_argument("--hsv-min-area", type=float, default=80.0)
    parser.add_argument("--hsv-roi", default="", help="x1,y1,x2,y2, empty = full image")

    return parser.parse_args()


# ==========================================================
# BASIC UTILS
# ==========================================================
def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_set_learning_mode(robot, enabled):
    try:
        robot.set_learning_mode(bool(enabled))
        print(f"[ROBOT] learning_mode={enabled}")
    except Exception as e:
        print("[WARN] set_learning_mode lỗi:", repr(e))


def safe_set_velocity(robot, velocity):
    for name in ["set_arm_max_velocity", "set_arm_max_velocity_percentage"]:
        fn = getattr(robot, name, None)
        if fn is None:
            continue

        try:
            fn(int(velocity))
            print(f"[ROBOT] {name}({velocity})")
            return
        except Exception as e:
            print(f"[WARN] {name} lỗi:", repr(e))


def get_joints(robot):
    t0 = time.perf_counter()
    joints = robot.get_joints()
    t1 = time.perf_counter()

    joints = np.asarray(joints, dtype=np.float32)
    latency_ms = (t1 - t0) * 1000.0

    return joints, latency_ms


def move_joints(robot, joints):
    robot.move_joints([float(x) for x in joints])


# ==========================================================
# CAMERA
# ==========================================================
def find_camera_index(args):
    fallback_idx = int(args.camera_index)
    found_idx = fallback_idx
    found_name = "UNKNOWN"

    for i in range(10):
        name_path = f"/sys/class/video4linux/video{i}/name"

        if not os.path.exists(name_path):
            continue

        try:
            with open(name_path, "r") as f:
                cam_name = f.read().strip()

            if args.camera_name_keyword in cam_name:
                found_idx = i
                found_name = cam_name
                break

        except Exception:
            pass

    print(f"[CAMERA] chọn /dev/video{found_idx} | name={found_name}")
    return found_idx, found_name


def open_camera(args):
    cam_idx, cam_name = find_camera_index(args)

    os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c auto_exposure=3 > /dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c brightness=0 > /dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c gain=50 > /dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c gamma=300 > /dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c contrast=40 > /dev/null 2>&1")
    os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c sharpness=45 > /dev/null 2>&1")

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera index {cam_idx}")

    return cap, cam_idx, cam_name


def resize_keep_aspect_bgr(frame_bgr, target_width=640):
    h, w = frame_bgr.shape[:2]
    scale = target_width / float(w)
    new_w = int(target_width)
    new_h = int(h * scale)

    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_frame(cap, args):
    ok, frame_bgr = cap.read()

    if not ok or frame_bgr is None:
        return None, None, None

    if args.flip_camera_horizontal:
        frame_bgr = cv2.flip(frame_bgr, 1)

    if args.flip_camera_vertical:
        frame_bgr = cv2.flip(frame_bgr, 0)

    frame_bgr = cv2.resize(
        frame_bgr,
        (args.camera_width, args.camera_height),
        interpolation=cv2.INTER_AREA,
    )

    display = frame_bgr.copy()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_224 = cv2.resize(
        frame_rgb,
        (args.image_size, args.image_size),
        interpolation=cv2.INTER_AREA,
    )

    raw_bgr_small = resize_keep_aspect_bgr(
        frame_bgr,
        target_width=args.raw_jpg_width,
    )

    return display, frame_224, raw_bgr_small


# ==========================================================
# HSV DEBUG ONLY
# ==========================================================
def parse_roi(roi_text, w, h):
    if not roi_text:
        return 0, 0, w, h

    vals = [int(v.strip()) for v in roi_text.split(",")]

    if len(vals) != 4:
        raise ValueError("--hsv-roi phải có dạng x1,y1,x2,y2")

    x1, y1, x2, y2 = vals

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return x1, y1, x2, y2


def detect_green_object(frame_bgr, args):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = parse_roi(args.hsv_roi, w, h)

    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < args.hsv_min_area:
            continue

        if area > best_area:
            best = contour
            best_area = area

    if best is None:
        feature = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return feature, None

    bx, by, bw, bh = cv2.boundingRect(best)
    bx += x1
    by += y1

    m = cv2.moments(best)

    if abs(m["m00"]) < 1e-6:
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
    else:
        cx = x1 + m["m10"] / m["m00"]
        cy = y1 + m["m01"] / m["m00"]

    cx_norm = float(cx / max(1, w))
    cy_norm = float(cy / max(1, h))
    area_norm = float(best_area / max(1, w * h))

    feature = np.array([cx_norm, cy_norm, area_norm], dtype=np.float32)
    box = (int(bx), int(by), int(bw), int(bh), int(cx), int(cy), float(best_area))

    return feature, box


def zone_from_cx(cx):
    if cx <= 0.0:
        return "unknown"
    if cx < 0.4:
        return "left"
    if cx <= 0.6:
        return "middle"
    return "right"


def draw_overlay(
    frame_bgr,
    args,
    status,
    saved_count,
    rec_points,
    traj_points,
    pending_points,
):
    out = frame_bgr.copy()

    feature, box = detect_green_object(frame_bgr, args)
    zone = zone_from_cx(float(feature[0]))

    cv2.rectangle(out, (0, 0), (out.shape[1], 120), (0, 0, 0), -1)

    cv2.putText(
        out,
        status,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    info = (
        f"saved_ep={saved_count} | rec={rec_points} | "
        f"traj={traj_points} | pending={pending_points}"
    )

    cv2.putText(
        out,
        info,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        out,
        f"HSV debug: cx={feature[0]:.3f} cy={feature[1]:.3f} area={feature[2]:.5f} zone={zone}",
        (10, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    cv2.putText(
        out,
        "T=start teach | P=stop teach | E=step replay | S=save | D=discard | Q=quit",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    if box is not None:
        x, y, bw, bh, cx, cy, area = box

        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)

    else:
        cv2.putText(
            out,
            "NO GREEN OBJECT",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if "RECORDING" in status:
        cv2.circle(out, (out.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(
            out,
            "REC",
            (out.shape[1] - 90, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

    return out, feature, zone


# ==========================================================
# TRAJECTORY PROCESSING
# ==========================================================
def validate_trajectory(traj, min_points=10, min_total_diff=0.02):
    if traj is None or len(traj) < min_points:
        print(f"[WARN] trajectory quá ngắn: {0 if traj is None else len(traj)} < {min_points}")
        return False

    traj = np.asarray(traj, dtype=np.float32)

    total_diff = float(np.linalg.norm(traj[-1] - traj[0]))
    step_norms = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    print("")
    print("[TRAJ] points:", len(traj))
    print("[TRAJ] first:", np.round(traj[0], 4).tolist())
    print("[TRAJ] last :", np.round(traj[-1], 4).tolist())
    print("[TRAJ] total_diff:", total_diff)
    print("[TRAJ] mean_step :", float(step_norms.mean()))
    print("[TRAJ] max_step  :", float(step_norms.max()))

    if total_diff < min_total_diff:
        print("[ERROR] trajectory gần như không có chuyển động.")
        return False

    return True


def remove_near_duplicate_joints(joints, min_dist=1e-4):
    joints = np.asarray(joints, dtype=np.float32)

    if len(joints) <= 1:
        return joints

    cleaned = [joints[0]]

    for i in range(1, len(joints)):
        dist = np.linalg.norm(joints[i] - cleaned[-1])

        if dist >= min_dist:
            cleaned.append(joints[i])

    return np.asarray(cleaned, dtype=np.float32)


def resample_by_joint_path(joints, num_points):
    """
    Resample theo độ dài đường đi trong không gian joint.

    Không dùng thời gian teach.
    Nếu teach được 30 điểm, có thể nội suy lên 100.
    Nếu teach được 150 điểm, có thể nén xuống 100.
    """
    joints = remove_near_duplicate_joints(joints)

    if len(joints) < 2:
        return joints

    step_dist = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step_dist)])

    total = cumulative[-1]

    if total < 1e-6:
        return joints

    num_points = max(2, int(num_points))
    new_s = np.linspace(0.0, total, num_points)

    resampled = []

    for j in range(joints.shape[1]):
        resampled_joint = np.interp(new_s, cumulative, joints[:, j])
        resampled.append(resampled_joint)

    resampled = np.stack(resampled, axis=1).astype(np.float32)

    return resampled


def make_actions_next_joints(joints):
    joints = np.asarray(joints, dtype=np.float32)

    actions = []

    for i in range(len(joints)):
        if i < len(joints) - 1:
            actions.append(joints[i + 1])
        else:
            actions.append(joints[i])

    return np.asarray(actions, dtype=np.float32)


def make_actions_delta_joints(joints):
    joints = np.asarray(joints, dtype=np.float32)

    actions = []

    for i in range(len(joints)):
        if i < len(joints) - 1:
            actions.append(joints[i + 1] - joints[i])
        else:
            actions.append(np.zeros_like(joints[i]))

    return np.asarray(actions, dtype=np.float32)


# ==========================================================
# DATASET SAVE
# ==========================================================
def get_next_episode_index(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

    files = [
        f for f in os.listdir(dataset_dir)
        if f.startswith("ep_") and f.endswith(".npz")
    ]

    if not files:
        return 0

    ids = []

    for f in files:
        try:
            ids.append(int(f.replace("ep_", "").replace(".npz", "")))
        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def save_raw_frames(raw_frames, dataset_dir, ep_index, raw_jpg_quality=92):
    raw_paths = []

    ep_dir = os.path.join(dataset_dir, "raw_frames", f"ep_{ep_index:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    for i, frame_bgr in enumerate(raw_frames):
        path = os.path.join(ep_dir, f"frame_{i:06d}.jpg")

        ok = cv2.imwrite(
            path,
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(raw_jpg_quality)],
        )

        if ok:
            raw_paths.append(path)
        else:
            raw_paths.append("")

    return raw_paths


def safe_save_episode(save_path, arrays):
    ensure_parent(save_path)

    tmp_path = save_path + ".tmp.npz"
    bak_path = save_path + ".bak.npz"

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if os.path.exists(save_path):
        try:
            shutil.copy2(save_path, bak_path)
            print("[BACKUP]", bak_path)
        except Exception as e:
            print("[WARN] backup lỗi:", repr(e))

    print("[SAFE_SAVE] writing tmp:", tmp_path)
    np.savez_compressed(tmp_path, **arrays)

    try:
        test = np.load(tmp_path, allow_pickle=True)
        print("[SAFE_SAVE] tmp check images:", test["images"].shape)
        print("[SAFE_SAVE] tmp check joints:", test["joints"].shape)
        print("[SAFE_SAVE] tmp check actions_delta_joints:", test["actions_delta_joints"].shape)
        test.close()
    except Exception as e:
        print("[ERROR] tmp file hỏng, không replace file chính:", repr(e))
        return False

    os.replace(tmp_path, save_path)

    print("")
    print("=" * 70)
    print("[SAVE_OK]", os.path.abspath(save_path))
    print("=" * 70)

    return True


def build_episode_arrays(
    ep_index,
    images,
    raw_image_paths,
    actual_joints,
    target_joints,
    timestamps,
    get_joints_latency_ms,
    instruction,
    taught_joints,
    taught_timestamps,
    replay_target_joints,
    start_object_feature,
    start_zone,
    success,
    note,
    args,
):
    images = np.asarray(images, dtype=np.uint8)
    actual_joints = np.asarray(actual_joints, dtype=np.float32)
    target_joints = np.asarray(target_joints, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    get_joints_latency_ms = np.asarray(get_joints_latency_ms, dtype=np.float32)

    taught_joints = np.asarray(taught_joints, dtype=np.float32)
    taught_timestamps = np.asarray(taught_timestamps, dtype=np.float64)
    replay_target_joints = np.asarray(replay_target_joints, dtype=np.float32)

    actions_next_joints = make_actions_next_joints(actual_joints)
    actions_delta_joints = make_actions_delta_joints(actual_joints)

    n = len(images)

    arrays = {
        # Main train fields
        "images": images,
        "joints": actual_joints,
        "actions_next_joints": actions_next_joints,
        "actions_delta_joints": actions_delta_joints,
        "instruction": np.asarray(instruction, dtype=object),

        # Variable-length episode metadata, không padding
        "valid_len": np.asarray(n, dtype=np.int32),
        "episode_id": np.asarray(ep_index, dtype=np.int32),
        "step_ids": np.arange(n, dtype=np.int32),
        "episode_ids": np.full((n,), ep_index, dtype=np.int32),

        # Debug / checking
        "target_joints": target_joints,
        "timestamps": timestamps,
        "get_joints_latency_ms": get_joints_latency_ms,
        "taught_joints": taught_joints,
        "taught_timestamps": taught_timestamps,
        "replay_target_joints": replay_target_joints,
        "raw_image_paths": np.asarray(raw_image_paths, dtype=object),

        # Object debug metadata, không dùng làm policy input chính
        "start_object_feature": np.asarray(start_object_feature, dtype=np.float32),
        "start_zone": np.asarray(start_zone, dtype=object),
        "object_text": np.asarray("green object", dtype=object),

        # Collector metadata
        "collector_version": np.asarray("v5_robot_frame_grid_step_replay_no_padding", dtype=object),
        "joint_source": np.asarray("step_replay_real_get_joints_single_connection", dtype=object),
        "action_source": np.asarray("delta_between_actual_step_joints", dtype=object),
        "resample_method": np.asarray("joint_path_arclength_not_time", dtype=object),

        "sample_hz_teach": np.asarray(args.teach_hz, dtype=np.float32),
        "replay_waypoints": np.asarray(args.replay_waypoints, dtype=np.int32),
        "camera_width": np.asarray(args.camera_width, dtype=np.int32),
        "camera_height": np.asarray(args.camera_height, dtype=np.int32),
        "image_size": np.asarray(args.image_size, dtype=np.int32),
        "velocity": np.asarray(args.velocity, dtype=np.int32),
        "pause_after_move": np.asarray(args.pause_after_move, dtype=np.float32),

        "success": np.asarray(bool(success)),
        "note": np.asarray(note, dtype=object),
    }

    return arrays


# ==========================================================
# STEP REPLAY
# ==========================================================
def execute_step_replay_and_capture(
    robot,
    cap,
    args,
    trajectory,
    saved_count,
    start_object_feature,
    start_zone,
):
    if not validate_trajectory(
        trajectory,
        min_points=args.min_points,
        min_total_diff=args.min_total_diff,
    ):
        return None

    replay_target_joints = resample_by_joint_path(
        trajectory,
        num_points=args.replay_waypoints,
    )

    if len(replay_target_joints) < 3:
        print("[ERROR] replay_target_joints quá ngắn.")
        return None

    print("")
    print("=" * 70)
    print("[E] STEP-REPLAY + CAPTURE")
    print("=" * 70)
    print("V4 sẽ đi từng waypoint, không chạy trajectory một phát.")
    print("Tại mỗi waypoint: move_joints -> get_joints thật -> chụp ảnh.")
    print("")
    print("[E] raw trajectory points:", len(trajectory))
    print("[E] replay waypoints    :", len(replay_target_joints))
    print("[E] instruction         :", args.instruction)
    print("=" * 70)

    safe_set_learning_mode(robot, False)

    try:
        print("[E] move to first waypoint...")
        move_joints(robot, replay_target_joints[0])
        time.sleep(args.move_start_delay)
    except Exception as e:
        print("[ERROR] không move tới waypoint đầu:", repr(e))
        return None

    images = []
    raw_frames = []
    actual_joints = []
    target_joints = []
    timestamps = []
    get_joints_latency_ms = []

    start_time = time.time()

    for i, target in enumerate(replay_target_joints):
        try:
            if i > 0:
                move_joints(robot, target)

            time.sleep(args.pause_after_move)

            cur_joints, latency_ms = get_joints(robot)

            display, frame_224, raw_bgr_small = read_frame(cap, args)

            if frame_224 is None:
                print("[WARN] frame None tại step", i)
                continue

            images.append(frame_224)
            raw_frames.append(raw_bgr_small)
            actual_joints.append(cur_joints)
            target_joints.append(np.asarray(target, dtype=np.float32))
            timestamps.append(time.time() - start_time)
            get_joints_latency_ms.append(latency_ms)

            show, _, _ = draw_overlay(
                display,
                args,
                f"EXEC STEP {i+1}/{len(replay_target_joints)}",
                saved_count,
                0,
                len(replay_target_joints),
                len(images),
            )

            cv2.imshow("TPES_V5", show)
            cv2.waitKey(1)

            print(
                f"[E] step {i+1:03d}/{len(replay_target_joints)} | "
                f"saved={len(images)} | "
                f"get_joints={latency_ms:.1f}ms | "
                f"target={np.round(target, 4).tolist()}"
            )

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Dừng step-replay.")
            break

        except Exception as e:
            print(f"[ERROR] lỗi tại step {i+1}:", repr(e))
            print("[ERROR] Hủy pending episode.")
            return None

    if len(images) < args.min_points:
        print(f"[WARN] pending quá ngắn: {len(images)} < {args.min_points}")
        return None

    actual_joints_arr = np.asarray(actual_joints, dtype=np.float32)
    actions_delta = make_actions_delta_joints(actual_joints_arr)
    action_norm = np.linalg.norm(actions_delta, axis=1)

    print("")
    print("=" * 70)
    print("[PENDING] Episode đã capture xong, chưa lưu.")
    print("[PENDING] images:", np.asarray(images).shape)
    print("[PENDING] joints:", actual_joints_arr.shape)
    print("[PENDING] actions_delta:", actions_delta.shape)
    print("[PENDING] action mean/max norm:", float(action_norm.mean()), float(action_norm.max()))
    print("[PENDING] start_object_feature:", start_object_feature.tolist(), "zone:", start_zone)
    print("[PENDING] Bấm S để lưu, D để bỏ.")
    print("=" * 70)

    return {
        "images": images,
        "raw_frames": raw_frames,
        "actual_joints": actual_joints,
        "target_joints": target_joints,
        "timestamps": timestamps,
        "get_joints_latency_ms": get_joints_latency_ms,
        "taught_joints": np.asarray(trajectory, dtype=np.float32),
        "replay_target_joints": replay_target_joints,
        "start_object_feature": np.asarray(start_object_feature, dtype=np.float32),
        "start_zone": str(start_zone),
        "trajectory_length": int(len(images)),
    }


# ==========================================================
# MAIN LOOP
# ==========================================================
def main():
    args = parse_args()

    print("")
    print("=" * 80)
    print("🔥 NIRYO miniVLA COLLECTOR V5 - ROBOT-FRAME GRID")
    print("=" * 80)
    print("Output dir       :", args.dataset_dir)
    print("Instruction      :", args.instruction)
    print("Robot zone       :", args.robot_zone)
    print("Replay waypoints :", args.replay_waypoints)
    print("No padding       : True")
    print("")
    print("Phím:")
    print("T = bắt đầu teach bằng tay, bật learning mode")
    print("P = dừng teach, tắt learning mode")
    print("E = step-replay + capture, tạo pending episode")
    print("S = save pending episode")
    print("D = discard pending/trajectory")
    print("L = learning mode ON")
    print("O = learning mode OFF")
    print("Q/ESC = thoát")
    print("=" * 80)

    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, "raw_frames"), exist_ok=True)

    cap, camera_index, camera_name = open_camera(args)
    cv2.namedWindow("TPES_V5", cv2.WINDOW_NORMAL)

    print("[ROBOT] connecting:", args.robot_ip)
    robot = NiryoRobot(args.robot_ip)
    safe_set_velocity(robot, args.velocity)
    safe_set_learning_mode(robot, True)

    saved_count = get_next_episode_index(args.dataset_dir)
    print("[DATASET] next episode index:", saved_count)

    recording = False
    trajectory = []
    trajectory_timestamps = []
    last_trajectory = None
    last_trajectory_timestamps = None
    pending_episode = None

    start_object_feature = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    start_zone = "unknown"

    last_t = 0.0
    teach_period = 1.0 / float(args.teach_hz)

    try:
        while True:
            display, _, _ = read_frame(cap, args)

            if display is None:
                continue

            if recording:
                status = "RECORDING TRAJECTORY: press P to stop"
            elif pending_episode is not None:
                status = "PENDING READY: press S to save, D to discard"
            elif last_trajectory is not None:
                status = "TRAJECTORY READY: reset object, press E"
            else:
                status = "READY: press T to teach"

            show, feature, zone = draw_overlay(
                display,
                args,
                status,
                saved_count,
                len(trajectory) if recording else 0,
                0 if last_trajectory is None else len(last_trajectory),
                0 if pending_episode is None else pending_episode["trajectory_length"],
            )

            cv2.imshow("TPES_V5", show)

            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord("q"), ord("Q")]:
                print("[QUIT] thoát.")
                break

            if key in [ord("l"), ord("L")]:
                safe_set_learning_mode(robot, True)
                continue

            if key in [ord("o"), ord("O")]:
                safe_set_learning_mode(robot, False)
                continue

            # ------------------------------
            # T = START TEACH
            # ------------------------------
            if key in [ord("t"), ord("T")] and not recording:
                print("")
                print("=" * 70)
                print(f"[T] START TEACH episode index {saved_count}")
                print("Kéo tay robot thực hiện thao tác. Bấm P để dừng.")
                print("[T] start_object_feature:", feature.tolist(), "zone:", zone)
                print("=" * 70)

                safe_set_learning_mode(robot, True)

                recording = True
                trajectory = []
                trajectory_timestamps = []
                last_trajectory = None
                last_trajectory_timestamps = None
                pending_episode = None

                start_object_feature = feature.copy()
                start_zone = zone

                last_t = 0.0
                time.sleep(0.15)
                continue

            # ------------------------------
            # RECORDING TEACH
            # ------------------------------
            if recording:
                now = time.time()

                if now - last_t >= teach_period:
                    try:
                        joints, latency_ms = get_joints(robot)
                        trajectory.append(joints)
                        trajectory_timestamps.append(now)

                        last_t = now

                        print(
                            f"[REC] points={len(trajectory)} | "
                            f"get_joints={latency_ms:.1f}ms",
                            end="\r",
                        )

                    except Exception as e:
                        print("\n[WARN] get_joints lỗi khi teach:", repr(e))

            # ------------------------------
            # P = STOP TEACH
            # ------------------------------
            if key in [ord("p"), ord("P")] and recording:
                print("")
                print("=" * 70)
                print(f"[P] STOP TEACH. Raw points={len(trajectory)}")
                print("[P] learning mode OFF")
                print("=" * 70)

                safe_set_learning_mode(robot, False)
                recording = False

                traj = np.asarray(trajectory, dtype=np.float32)
                traj_t = np.asarray(trajectory_timestamps, dtype=np.float64)

                if validate_trajectory(
                    traj,
                    min_points=args.min_points,
                    min_total_diff=args.min_total_diff,
                ):
                    last_trajectory = traj
                    last_trajectory_timestamps = traj_t
                    pending_episode = None

                    print("[P] trajectory saved in RAM.")
                    print("[NEXT] Đặt lại vật về vị trí ban đầu, bỏ tay ra, rồi bấm E.")

                else:
                    last_trajectory = None
                    last_trajectory_timestamps = None
                    pending_episode = None

                    print("[P] trajectory lỗi, bấm T thu lại.")

                trajectory = []
                trajectory_timestamps = []
                time.sleep(0.25)
                continue

            # ------------------------------
            # E = STEP REPLAY + CAPTURE
            # ------------------------------
            if key in [ord("e"), ord("E")] and not recording:
                if last_trajectory is None:
                    print("[WARN] chưa có trajectory. Bấm T rồi P trước.")
                    continue

                pending_episode = execute_step_replay_and_capture(
                    robot=robot,
                    cap=cap,
                    args=args,
                    trajectory=last_trajectory,
                    saved_count=saved_count,
                    start_object_feature=start_object_feature,
                    start_zone=start_zone,
                )

                if pending_episode is None:
                    print("[E] execute/capture lỗi. Có thể bấm E lại hoặc D bỏ.")
                else:
                    print("[E] pending episode ready. Bấm S để lưu, D để bỏ.")

                continue

            # ------------------------------
            # S = SAVE PENDING EPISODE
            # ------------------------------
            if key in [ord("s"), ord("S")] and not recording:
                if pending_episode is None:
                    print("[WARN] chưa có pending episode. Bấm E trước.")
                    continue

                success_input = input("\n✅ Episode này robot làm đúng task không? [Y/n]: ").strip().lower()
                success = success_input not in ["n", "no", "0", "false"]

                object_moved_input = input("📦 Vật có bị đẩy đúng hướng không? [Y/n]: ").strip().lower()
                object_moved = object_moved_input not in ["n", "no", "0", "false"]

                table_touch_input = input("⚠️ Gripper có chạm/cạ bàn không? [y/N]: ").strip().lower()
                table_touch = table_touch_input in ["y", "yes", "1", "true"]

                note = input("📝 Ghi chú ngắn, có thể bỏ trống: ").strip()

                ep_index = saved_count

                raw_image_paths = save_raw_frames(
                    raw_frames=pending_episode["raw_frames"],
                    dataset_dir=args.dataset_dir,
                    ep_index=ep_index,
                    raw_jpg_quality=92,
                )

                arrays = build_episode_arrays(
                    ep_index=ep_index,
                    images=pending_episode["images"],
                    raw_image_paths=raw_image_paths,
                    actual_joints=pending_episode["actual_joints"],
                    target_joints=pending_episode["target_joints"],
                    timestamps=pending_episode["timestamps"],
                    get_joints_latency_ms=pending_episode["get_joints_latency_ms"],
                    instruction=args.instruction,
                    taught_joints=pending_episode["taught_joints"],
                    taught_timestamps=last_trajectory_timestamps,
                    replay_target_joints=pending_episode["replay_target_joints"],
                    start_object_feature=pending_episode["start_object_feature"],
                    start_zone=pending_episode["start_zone"],
                    success=success,
                    note=note,
                    args=args,
                )

                # ==============================
                # V5 robot-frame metadata
                # ==============================
                robot_zone = str(args.robot_zone)

                if "_" in robot_zone:
                    robot_distance_bin, robot_lateral_bin = robot_zone.split("_", 1)
                else:
                    robot_distance_bin, robot_lateral_bin = "unknown", "unknown"

                arrays["robot_zone"] = np.asarray(robot_zone, dtype=object)
                arrays["robot_distance_bin"] = np.asarray(robot_distance_bin, dtype=object)
                arrays["robot_lateral_bin"] = np.asarray(robot_lateral_bin, dtype=object)
                arrays["object_moved"] = np.asarray(bool(object_moved))
                arrays["table_touch"] = np.asarray(bool(table_touch))
                arrays["data_version"] = np.asarray("v5_robot_frame_grid", dtype=object)

                save_path = os.path.join(
                    args.dataset_dir,
                    f"ep_{ep_index:03d}.npz",
                )

                ok = safe_save_episode(save_path, arrays)

                if ok:
                    print(f"[S] saved episode ep_{ep_index:03d}.npz")
                    print("images:", arrays["images"].shape, arrays["images"].dtype)
                    print("joints:", arrays["joints"].shape, arrays["joints"].dtype)
                    print("actions_delta_joints:", arrays["actions_delta_joints"].shape, arrays["actions_delta_joints"].dtype)
                    print("valid_len:", int(arrays["valid_len"]))

                    saved_count += 1
                    pending_episode = None
                    last_trajectory = None
                    last_trajectory_timestamps = None
                    trajectory = []
                    trajectory_timestamps = []

                    safe_set_learning_mode(robot, True)

                else:
                    print("[S] save lỗi. Không xóa pending episode.")

                continue

            # ------------------------------
            # D = DISCARD
            # ------------------------------
            if key in [ord("d"), ord("D")] and not recording:
                print("[D] discard pending and trajectory")

                pending_episode = None
                last_trajectory = None
                last_trajectory_timestamps = None
                trajectory = []
                trajectory_timestamps = []

                safe_set_learning_mode(robot, True)
                continue

    except KeyboardInterrupt:
        print("\n[INTERRUPT] thoát.")

    finally:
        try:
            safe_set_learning_mode(robot, False)
        except Exception:
            pass

        try:
            robot.close_connection()
        except Exception:
            pass

        try:
            cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
