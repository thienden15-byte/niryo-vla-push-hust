#!/usr/bin/env python3
import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    from pyniryo import NiryoRobot
except Exception as e:
    print("❌ Không import được pyniryo:", e)
    raise SystemExit(1)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_goal_v2_cluster_on_niryo import predict_cluster


def normalize_color_name(s):
    s = str(s).strip().lower()

    mapping = {
        "1": "green",
        "g": "green",
        "green": "green",
        "xanh": "green",
        "xanh la": "green",
        "xanh lá": "green",
        "xanh_la": "green",
        "xanh_lá": "green",

        "2": "red",
        "r": "red",
        "red": "red",
        "do": "red",
        "đỏ": "red",
        "mau do": "red",
        "màu đỏ": "red",

        "3": "blue",
        "b": "blue",
        "blue": "blue",
        "xanh duong": "blue",
        "xanh dương": "blue",
        "xanh_duong": "blue",
        "xanh_dương": "blue",
    }

    return mapping.get(s, None)


def ask_color_from_terminal():
    print("\nChọn màu vật cần đẩy:")
    print("  1 / green / xanh lá     : vật xanh lá")
    print("  2 / red   / đỏ          : vật đỏ")
    print("  3 / blue  / xanh dương  : vật xanh dương")

    while True:
        ans = input("Nhập màu vật cần đẩy: ").strip()
        color = normalize_color_name(ans)

        if color is not None:
            print(f"Đã chọn màu: {color}")
            return color

        print("Không hiểu màu này. Nhập lại: green / red / blue hoặc 1 / 2 / 3.")


def detect_auto_color(img224, args):
    h_img, w_img = img224.shape[:2]

    x1 = int(args.roi_x1 * w_img)
    x2 = int(args.roi_x2 * w_img)
    y1 = int(args.roi_y1 * h_img)
    y2 = int(args.roi_y2 * h_img)

    hsv = cv2.cvtColor(img224, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    color = args.color.lower()

    if color == "green":
        color_mask = (
            (h >= args.green_h_min) &
            (h <= args.green_h_max) &
            (s >= args.s_min) &
            (v >= args.v_min) &
            (v <= args.v_max)
        )
    elif color == "blue":
        color_mask = (
            (h >= args.blue_h_min) &
            (h <= args.blue_h_max) &
            (s >= args.s_min) &
            (v >= args.v_min) &
            (v <= args.v_max)
        )
    elif color == "red":
        color_mask = (
            (
                (h >= args.red_h1_min) &
                (h <= args.red_h1_max)
            )
            |
            (
                (h >= args.red_h2_min) &
                (h <= args.red_h2_max)
            )
        ) & (s >= args.s_min) & (v >= args.v_min) & (v <= args.v_max)
    else:
        raise ValueError(f"Unsupported color: {args.color}")

    mask = color_mask.astype(np.uint8) * 255

    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    k_open = np.ones((3, 3), np.uint8)
    k_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = float(cv2.contourArea(c))

        if area < args.min_area_px:
            continue

        if area > args.max_area_px:
            continue

        x, y, ww, hh = cv2.boundingRect(c)

        if ww <= 1 or hh <= 1:
            continue

        ratio = max(ww / hh, hh / ww)
        if ratio > args.max_ratio:
            continue

        M = cv2.moments(c)
        if M["m00"] > 1e-6:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = x + ww / 2.0
            cy = y + hh / 2.0

        # Ưu tiên vật nằm bên phải trong ROI và diện tích gần vật thật.
        right_score = 1000.0 * (cx / float(w_img))
        area_score = -abs(area - args.target_area_px)

        score = right_score + area_score

        candidates.append((score, area, c, cx, cy, x, y, ww, hh))

    if not candidates:
        return None, mask

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, area, c, cx, cy, x, y, ww, hh = candidates[0]

    cx_norm = cx / float(w_img)
    cy_norm = cy / float(h_img)
    area_norm = area / float(w_img * h_img)

    return {
        "feature": np.array([cx_norm, cy_norm, area_norm], dtype=np.float32),
        "contour": c,
        "cx": cx,
        "cy": cy,
        "bbox": (x, y, ww, hh),
        "area": area,
    }, mask


def execute_robot_thread(args, object_feature, robot_state):
    robot_state["running"] = True
    robot_state["done"] = False
    robot_state["error"] = None

    try:
        print("\n" + "=" * 100)
        print("LOCKED object_feature:", object_feature)
        print("Đang sinh Goal V2 15 điểm...")

        policy = np.load(args.policy, allow_pickle=True)

        cluster_points, info = predict_cluster(
            policy,
            object_feature=object_feature,
            k=args.k,
            power=args.power,
        )

        print("\nAI Goal V2 result:")
        print("Object feature:", object_feature)
        print("Cluster idxs:", info["cluster_idxs"].tolist())
        print("Predicted cluster points:", cluster_points.shape)

        print("\nSelected demos:")
        for f, d, w in zip(info["selected_files"], info["selected_ds"], info["selected_weights"]):
            print(f"  {str(f):12s} dist={float(d):.6f} weight={float(w):.3f}")

        path = cluster_points.astype(np.float32)

        print("\nMode: RAW 15 POINTS - gửi đúng 15 điểm AI sinh ra")
        print("Path sent to PyNiryo:", path.shape)
        print("First:", np.array2string(path[0], precision=6, suppress_small=True))
        print("Last :", np.array2string(path[-1], precision=6, suppress_small=True))

        if args.dry_run:
            print("\nDRY RUN: chỉ detect + sinh 15 điểm, không chạy robot.")
            return

        print("\n⚠️ Robot thật chuẩn bị chạy. Tay gần nút dừng khẩn cấp.")
        print("Kết nối robot...")

        robot = NiryoRobot(args.ip)

        try:
            robot.set_learning_mode(False)
            robot.set_arm_max_velocity(args.velocity)

            current_joints = np.array(robot.get_joints(), dtype=np.float32)
            print("Current joints:", np.array2string(current_joints, precision=6, suppress_small=True))

            list_joints = [q.tolist() for q in path]

            print("\nExecuting trajectory on real robot...")
            try:
                robot.execute_trajectory_from_poses_and_joints(
                    list_joints,
                    ["joint"] * len(list_joints)
                )
            except Exception as e:
                print("⚠️ execute_trajectory lỗi, fallback move_joints từng điểm.")
                print("Lỗi:", e)
                for i, q in enumerate(list_joints):
                    print(f"Waypoint {i+1}/{len(list_joints)}")
                    robot.move_joints(q)
                    time.sleep(0.05)

            print("\n✅ DONE: robot đã chạy xong.")

        finally:
            try:
                robot.close_connection()
            except Exception:
                pass

    except Exception as e:
        robot_state["error"] = str(e)
        print("\n❌ Lỗi robot thread:", e)

    finally:
        robot_state["running"] = False
        robot_state["done"] = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--camera-index", type=int, default=3)
    parser.add_argument("--policy", type=str, default="goal_v1/checkpoints/goal_v2_cluster_policy.npz")

    parser.add_argument(
        "--color",
        type=str,
        default="ask",
        help="Màu vật cần đẩy: green/red/blue. Nếu để ask thì chương trình sẽ hỏi khi chạy.",
    )

    parser.add_argument("--velocity", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--power", type=float, default=1.0)

    parser.add_argument("--green-h-min", type=int, default=35)
    parser.add_argument("--green-h-max", type=int, default=90)

    parser.add_argument("--blue-h-min", type=int, default=90)
    parser.add_argument("--blue-h-max", type=int, default=130)

    parser.add_argument("--red-h1-min", type=int, default=0)
    parser.add_argument("--red-h1-max", type=int, default=12)
    parser.add_argument("--red-h2-min", type=int, default=165)
    parser.add_argument("--red-h2-max", type=int, default=179)

    parser.add_argument("--s-min", type=int, default=25)
    parser.add_argument("--v-min", type=int, default=40)
    parser.add_argument("--v-max", type=int, default=255)

    parser.add_argument("--min-area-px", type=float, default=15.0)
    parser.add_argument("--max-area-px", type=float, default=700.0)
    parser.add_argument("--target-area-px", type=float, default=250.0)
    parser.add_argument("--max-ratio", type=float, default=4.0)

    parser.add_argument("--roi-x1", type=float, default=0.50)
    parser.add_argument("--roi-x2", type=float, default=0.95)
    parser.add_argument("--roi-y1", type=float, default=0.05)
    parser.add_argument("--roi-y2", type=float, default=0.95)

    parser.add_argument("--display-scale", type=float, default=1.0)

    parser.add_argument(
        "--no-raw-view",
        action="store_true",
        help="Tắt cửa sổ camera raw to.",
    )
    parser.add_argument(
        "--raw-display-width",
        type=int,
        default=1000,
        help="Chiều rộng cửa sổ camera raw. Ví dụ 900, 1000, 1200.",
    )

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.color == "ask":
        args.color = ask_color_from_terminal()
    else:
        normalized = normalize_color_name(args.color)
        if normalized is None:
            raise ValueError(f"Không hiểu màu: {args.color}. Dùng green/red/blue.")
        args.color = normalized

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print("❌ Không thấy policy:", policy_path)
        raise SystemExit(1)

    print("=" * 100)
    print("AUTO COLOR CAMERA → GOAL V2 CLUSTER → REAL NIRYO")
    print("=" * 100)
    print("Policy:", policy_path)
    print("Robot IP:", args.ip)
    print("Camera index:", args.camera_index)
    print("Target color:", args.color)
    print("Trajectory mode: RAW 15 POINTS")
    print("SPACE = lock object + start robot thread")
    print("q     = thoát camera")
    print("=" * 100)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Không mở được camera index:", args.camera_index)
        raise SystemExit(1)

    Path("goal_v1/logs").mkdir(parents=True, exist_ok=True)

    win = "LIVE CAMERA | SPACE = RUN | q = QUIT"
    cv2.namedWindow(win)

    win_raw = "RAW CAMERA BIG VIEW"
    if not args.no_raw_view:
        cv2.namedWindow(win_raw, cv2.WINDOW_NORMAL)

    robot_state = {
        "running": False,
        "done": False,
        "error": None,
    }
    robot_thread = None
    last_feature = None
    locked_feature = None

    print("\nCamera đang chạy.")
    print("Nếu khung xanh/chấm đỏ đúng vật, bấm SPACE.")
    print("Sau khi SPACE, camera vẫn tiếp tục mở.")
    print("Robot chạy xong, bấm q để thoát camera.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Không đọc được frame.")
            time.sleep(0.03)
            continue

        img224 = cv2.resize(frame, (224, 224))
        vis = img224.copy()

        result, raw_mask = detect_auto_color(img224, args)
        display_mask = np.zeros_like(raw_mask)

        h_img, w_img = img224.shape[:2]

        rx1 = int(args.roi_x1 * w_img)
        rx2 = int(args.roi_x2 * w_img)
        ry1 = int(args.roi_y1 * h_img)
        ry2 = int(args.roi_y2 * h_img)

        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        if result is not None:
            feature = result["feature"]
            x, y, ww, hh = result["bbox"]
            cx = result["cx"]
            cy = result["cy"]

            last_feature = feature

            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.drawContours(display_mask, [result["contour"]], -1, 255, thickness=-1)

            txt1 = f"{args.color} cx={feature[0]:.3f} cy={feature[1]:.3f} area={feature[2]:.5f}"
            txt2 = f"area_px={result['area']:.1f}"
            cv2.putText(vis, txt1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            cv2.putText(vis, txt2, (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        else:
            last_feature = None
            cv2.putText(vis, f"NO {args.color.upper()} OBJECT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

        if locked_feature is not None:
            cv2.putText(vis, "LOCKED - ROBOT STARTED", (5, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if robot_state["running"]:
            cv2.putText(vis, "ROBOT RUNNING - CAMERA STILL ON", (5, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 2)
        elif robot_state["done"]:
            cv2.putText(vis, "ROBOT DONE - press q to quit", (5, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 2)
        else:
            cv2.putText(vis, "SPACE = RUN | q = CANCEL", (5, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        mask_bgr = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
        show = np.hstack([vis, mask_bgr])

        if args.display_scale != 1.0:
            show = cv2.resize(
                show,
                None,
                fx=args.display_scale,
                fy=args.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imshow(win, show)

        # Cửa sổ raw camera to: hiển thị ảnh thật camera đang thấy, không HSV, không mask.
        if not args.no_raw_view:
            raw = frame.copy()
            raw_h, raw_w = raw.shape[:2]

            target_w = int(args.raw_display_width)
            scale = target_w / float(raw_w)
            target_h = int(raw_h * scale)

            raw_show = cv2.resize(
                raw,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )

            cv2.resizeWindow(win_raw, target_w, target_h)
            cv2.imshow(win_raw, raw_show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            if robot_state["running"]:
                print("⚠️ Camera sẽ đóng, nhưng robot thread có thể vẫn đang chạy. Muốn dừng robot thì dùng nút dừng khẩn cấp.")
            break

        if key == 32:
            if robot_state["running"]:
                print("⚠️ Robot đang chạy rồi, bỏ qua SPACE.")
                continue

            if robot_state["done"]:
                print("Robot đã chạy xong rồi. Bấm q để thoát hoặc chạy lại script.")
                continue

            if last_feature is None:
                print("⚠️ SPACE nhưng chưa detect được vật. Không chạy robot.")
                continue

            locked_feature = last_feature.copy()

            cv2.imwrite("goal_v1/logs/live_v2_locked_frame.png", show)
            print("Saved locked frame: goal_v1/logs/live_v2_locked_frame.png")
            print("Locked object_feature:", locked_feature)

            robot_thread = threading.Thread(
                target=execute_robot_thread,
                args=(args, locked_feature, robot_state),
                daemon=True,
            )
            robot_thread.start()

    cap.release()
    cv2.destroyAllWindows()

    if robot_thread is not None and robot_thread.is_alive():
        print("Đợi robot thread kết thúc...")
        robot_thread.join()

    print("Đã thoát chương trình.")


if __name__ == "__main__":
    main()
