#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

try:
    from pyniryo import NiryoRobot
except Exception as e:
    print("❌ Không import được pyniryo:", e)
    raise SystemExit(1)


def majority_vote(labels):
    labels = [str(x) for x in labels]
    c = Counter(labels)
    max_count = max(c.values())
    candidates = sorted([k for k, v in c.items() if v == max_count])
    return candidates[0]


def interp_goal(train_feat_pool, goal_pool, x, k=5, power=1.0):
    n = len(train_feat_pool)
    k = min(k, n)

    d = np.linalg.norm(train_feat_pool - x[None, :], axis=1)
    order = np.argsort(d)
    ids = order[:k]
    ds = d[ids]

    if ds[0] < 1e-9:
        pred = goal_pool[ids[0]]
        weights = np.zeros_like(ds)
        weights[0] = 1.0
    else:
        weights = 1.0 / ((ds + 1e-6) ** power)
        weights = weights / weights.sum()
        pred = (goal_pool[ids] * weights[:, None]).sum(axis=0)

    return pred.astype(np.float32), ids, ds.astype(np.float32), weights.astype(np.float32)


def predict_goal18(policy, object_feature):
    train_goals = policy["train_goals"].astype(np.float32)
    train_cy_area_n = policy["train_cy_area_n"].astype(np.float32)
    train_zones = np.array([str(x) for x in policy["train_robot_zones"]])
    train_files = np.array([str(x) for x in policy["train_episode_files"]])

    cy_area_mean = policy["cy_area_mean"].astype(np.float32)
    cy_area_std = policy["cy_area_std"].astype(np.float32)

    classifier_k = int(policy["classifier_k"])
    zone_interp_k = int(policy["zone_interp_k"])
    power = float(policy["interp_power"])

    cy_area = object_feature[[1, 2]].astype(np.float32)
    x = (cy_area - cy_area_mean) / cy_area_std

    d = np.linalg.norm(train_cy_area_n - x[None, :], axis=1)
    order = np.argsort(d)

    cls_ids = order[:classifier_k]
    cls_labels = train_zones[cls_ids]
    pred_zone = majority_vote(cls_labels)
    agree = sum(str(l) == pred_zone for l in cls_labels) / len(cls_labels)

    pool_pos = np.where(train_zones == pred_zone)[0]
    if len(pool_pos) == 0:
        pool_pos = np.arange(len(train_goals))

    goal18, ids_in_pool, ds, weights = interp_goal(
        train_feat_pool=train_cy_area_n[pool_pos],
        goal_pool=train_goals[pool_pos],
        x=x,
        k=zone_interp_k,
        power=power,
    )

    selected_pos = pool_pos[ids_in_pool]

    info = {
        "pred_zone": pred_zone,
        "agree": agree,
        "selected_files": train_files[selected_pos],
        "selected_zones": train_zones[selected_pos],
        "selected_ds": ds,
        "selected_weights": weights,
    }

    return goal18, info


def build_smooth_path(start_joints, goal18, max_step_rad=0.025):
    goals = goal18.reshape(3, 6).astype(np.float32)

    waypoints = [start_joints.astype(np.float32)]
    current = start_joints.astype(np.float32)

    for target in goals:
        diff = target - current
        max_abs = float(np.max(np.abs(diff)))
        steps = max(2, int(np.ceil(max_abs / max_step_rad)) + 1)

        for s in range(1, steps + 1):
            alpha = s / steps
            q = current * (1.0 - alpha) + target * alpha
            waypoints.append(q.astype(np.float32))

        current = target

    return np.stack(waypoints, axis=0)


clicked = None


def mouse_cb(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (x, y)
        print(f"Clicked object at x={x}, y={y}")


def get_hsv_range_from_click(img224, click_xy, h_margin=10, s_margin=70, v_margin=70):
    x, y = click_xy

    x1 = max(0, x - 4)
    x2 = min(img224.shape[1], x + 5)
    y1 = max(0, y - 4)
    y2 = min(img224.shape[0], y + 5)

    patch = img224[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h = int(np.median(hsv[:, :, 0]))
    s = int(np.median(hsv[:, :, 1]))
    v = int(np.median(hsv[:, :, 2]))

    lower = np.array([max(0, h - h_margin), max(0, s - s_margin), max(0, v - v_margin)], dtype=np.uint8)
    upper = np.array([min(179, h + h_margin), min(255, s + s_margin), min(255, v + v_margin)], dtype=np.uint8)

    return lower, upper, (h, s, v)


def detect_object_feature(cap, args):
    global clicked
    clicked = None

    win = "Click object, then press SPACE to accept"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)

    hsv_lower = None
    hsv_upper = None
    last_feature = None
    last_vis = None

    print("\nCamera mode:")
    print("- Click vào vật cần đẩy.")
    print("- Sau khi thấy khung xanh đúng vật, nhấn SPACE.")
    print("- Nhấn q để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Không đọc được camera frame.")
            continue

        img224 = cv2.resize(frame, (224, 224))

        if clicked is not None and hsv_lower is None:
            hsv_lower, hsv_upper, hsv_center = get_hsv_range_from_click(
                img224,
                clicked,
                h_margin=args.h_margin,
                s_margin=args.s_margin,
                v_margin=args.v_margin,
            )
            print("HSV center:", hsv_center)
            print("HSV lower :", hsv_lower)
            print("HSV upper :", hsv_upper)

        vis = img224.copy()

        if hsv_lower is not None:
            hsv = cv2.cvtColor(img224, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours and clicked is not None:
                click_x, click_y = clicked

                candidates = []

                for c in contours:
                    area = float(cv2.contourArea(c))
                    if area < args.min_area_px:
                        continue

                    M = cv2.moments(c)
                    if M["m00"] <= 1e-6:
                        continue

                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])

                    dist_to_click = float(((cx - click_x) ** 2 + (cy - click_y) ** 2) ** 0.5)

                    # Ưu tiên vùng gần điểm click.
                    # Có phạt nhẹ vùng quá nhỏ, nhưng không chọn vùng lớn nhất toàn ảnh nữa.
                    score = dist_to_click - 0.01 * area

                    candidates.append((score, dist_to_click, area, c, cx, cy))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    score, dist_to_click, area, c, cx, cy = candidates[0]

                    cx_norm = cx / 224.0
                    cy_norm = cy / 224.0
                    area_norm = area / float(224 * 224)

                    last_feature = np.array([cx_norm, cy_norm, area_norm], dtype=np.float32)

                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    cv2.circle(vis, (int(click_x), int(click_y)), 4, (255, 0, 0), -1)
                    cv2.line(vis, (int(click_x), int(click_y)), (int(cx), int(cy)), (255, 255, 0), 1)

                    txt1 = f"cx={cx_norm:.3f} cy={cy_norm:.3f} area={area_norm:.5f}"
                    txt2 = f"dist_click={dist_to_click:.1f}px"
                    cv2.putText(vis, txt1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    cv2.putText(vis, txt2, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if clicked is not None:
            cv2.circle(vis, clicked, 4, (255, 0, 0), -1)

        cv2.imshow(win, vis)
        last_vis = vis.copy()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.destroyWindow(win)
            return None

        if key == ord("r"):
            clicked = None
            hsv_lower = None
            hsv_upper = None
            last_feature = None
            print("Reset click.")

        if key == 32:
            if last_feature is None:
                print("⚠️ Chưa detect được vật. Click lại hoặc chỉnh ánh sáng.")
                continue

            cv2.imwrite("goal_v1/logs/live_detect_last.png", last_vis)
            cv2.destroyWindow(win)
            return last_feature


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--camera-index", type=int, default=3)
    parser.add_argument("--policy", type=str, default="goal_v1/checkpoints/goal_v1_retrieval_policy.npz")
    parser.add_argument("--velocity", type=int, default=3)
    parser.add_argument("--max-step-rad", type=float, default=0.025)

    parser.add_argument("--h-margin", type=int, default=10)
    parser.add_argument("--s-margin", type=int, default=70)
    parser.add_argument("--v-margin", type=int, default=70)
    parser.add_argument("--min-area-px", type=float, default=20.0)

    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print("❌ Không thấy policy:", policy_path)
        raise SystemExit(1)

    print("=" * 100)
    print("LIVE CAMERA → GOAL_V1 → REAL NIRYO")
    print("=" * 100)
    print("Policy:", policy_path)
    print("Robot IP:", args.ip)
    print("Camera index:", args.camera_index)
    print("=" * 100)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Không mở được camera index:", args.camera_index)
        raise SystemExit(1)

    object_feature = detect_object_feature(cap, args)
    cap.release()

    if object_feature is None:
        print("Đã hủy camera detection.")
        return

    print("\nDetected object_feature [cx, cy, area]:", object_feature)

    policy = np.load(policy_path, allow_pickle=True)
    goal18, info = predict_goal18(policy, object_feature)

    print("\nAI goal_v1 result:")
    print("Pred zone:", info["pred_zone"])
    print("Agree:", info["agree"])
    print("Selected demos:")
    for f, z, d, w in zip(info["selected_files"], info["selected_zones"], info["selected_ds"], info["selected_weights"]):
        print(f"  {str(f):12s} zone={str(z):14s} dist={float(d):.6f} weight={float(w):.3f}")

    goals = goal18.reshape(3, 6)
    print("\nPredicted joint goals:")
    for i in range(3):
        print(f"  goal {i+1}:", np.array2string(goals[i], precision=6, suppress_small=True))

    print("\n⚠️ Chuẩn bị chạy ROBOT THẬT.")
    print("Kiểm tra:")
    print("1. Vật đang ở đúng vùng camera vừa detect.")
    print("2. Không có tay/người trong vùng robot.")
    print("3. Tốc độ thấp.")
    print("4. Tay gần nút dừng khẩn cấp.")

    ans = input("\nGõ YES_RUN để robot thật chạy: ").strip()
    if ans != "YES_RUN":
        print("Đã hủy, robot không chạy.")
        return

    robot = None

    try:
        print("\nKết nối robot...")
        robot = NiryoRobot(args.ip)
        robot.set_learning_mode(False)
        robot.set_arm_max_velocity(args.velocity)

        current_joints = np.array(robot.get_joints(), dtype=np.float32)
        print("Current joints:", np.array2string(current_joints, precision=6, suppress_small=True))

        path = build_smooth_path(
            start_joints=current_joints,
            goal18=goal18,
            max_step_rad=args.max_step_rad,
        )

        print("Generated path:", path.shape)
        print("First:", np.array2string(path[0], precision=6, suppress_small=True))
        print("Last :", np.array2string(path[-1], precision=6, suppress_small=True))

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
                time.sleep(0.03)

        print("\n✅ DONE: robot thật đã chạy theo object_feature camera.")

    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C. Nếu robot còn chạy, bấm dừng khẩn cấp.")

    except Exception as e:
        print("\n❌ Lỗi:", e)

    finally:
        if robot is not None:
            try:
                robot.close_connection()
            except Exception:
                pass


if __name__ == "__main__":
    main()
