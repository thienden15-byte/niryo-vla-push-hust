import os, sys, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser("~/TinyVLA"))
sys.path.insert(0, os.path.expanduser("~/TinyVLA/llava-pythia"))

diffik_path = Path.home() / "tinyvla_niryo_runtime/scripts/run_author_style_niryo_diffik_rollout.py"
spec = importlib.util.spec_from_file_location("diffik_runtime", diffik_path)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def detect_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 45, 45], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 400:
        return None, mask

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w / 2
    cy = y + h / 2

    return {
        "bbox": (x, y, w, h),
        "center": (cx, cy),
        "area": area,
    }, mask


def capture_object_and_compute(args):
    cam_idx, cam_name, candidates = R.base_live.find_usb_camera()
    print("camera candidates:", candidates)
    print("auto selected:", cam_idx, cam_name)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(25):
        ok, frame = cap.read()
        time.sleep(0.03)

    cap.release()

    if frame is None:
        raise RuntimeError("Cannot capture frame")

    green, mask = detect_green(frame)
    if green is None:
        raise RuntimeError("Green object not detected")

    x, y, w, h = green["bbox"]
    cx, cy = green["center"]
    area = green["area"]

    # Tính vị trí robot cần tiếp xúc tương đối từ ảnh.
    contact_x = args.ref_x + (cx - args.ref_cx) * args.px_to_x
    contact_y = args.ref_y + args.y_sign * (cy - args.ref_cy) * args.px_to_y

    contact_x = clamp(contact_x + args.tool_x_offset, args.min_contact_x, args.max_contact_x)
    contact_y = clamp(contact_y + args.tool_y_offset, -args.max_abs_y, args.max_abs_y)

    # Điểm trước vật: robot hạ thấp ở đây trước, rồi mới push vào vật.
    precontact_x = contact_x - args.precontact_offset
    precontact_x = clamp(precontact_x, args.min_precontact_x, args.max_contact_x)

    print()
    print("===== GREEN OBJECT DETECTED =====")
    print("bbox x,y,w,h:", (x, y, w, h))
    print("center cx,cy:", (round(cx, 2), round(cy, 2)))
    print("area:", round(area, 2))

    print()
    print("===== AUTO TARGETS =====")
    print("y_sign:", args.y_sign)
    print("contact_x     :", round(contact_x, 6))
    print("precontact_x  :", round(precontact_x, 6))
    print("target_y      :", round(contact_y, 6))
    print("contact_z     :", round(args.contact_z, 6))
    print("precontact gap:", round(args.precontact_offset, 6))

    debug = frame.copy()
    cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.circle(debug, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    cv2.putText(
        debug,
        f"green center=({cx:.1f},{cy:.1f})",
        (x, max(30, y-15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        debug,
        f"pre_x={precontact_x:.3f} contact_x={contact_x:.3f} y={contact_y:.3f} z={args.contact_z:.3f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    out_dir = Path.home() / "tinyvla_niryo_runtime/auto_green_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = out_dir / f"auto_green_precontact_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(debug_path), debug)
    print("saved debug image:", debug_path)

    return precontact_x, contact_x, contact_y, cam_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")

    # Reference từ lần vật ở giữa đẩy được.
    p.add_argument("--ref-cx", type=float, default=749.0)
    p.add_argument("--ref-cy", type=float, default=413.0)
    p.add_argument("--ref-x", type=float, default=0.300)
    p.add_argument("--ref-y", type=float, default=0.000)

    # Pixel -> robot rough mapping.
    p.add_argument("--px-to-x", type=float, default=0.00018)
    p.add_argument("--px-to-y", type=float, default=0.00032)
    p.add_argument("--y-sign", type=float, default=-1.0)

    p.add_argument("--min-contact-x", type=float, default=0.245)
    p.add_argument("--max-contact-x", type=float, default=0.325)
    p.add_argument("--min-precontact-x", type=float, default=0.180)
    p.add_argument("--tool-x-offset", type=float, default=0.000)
    p.add_argument("--tool-y-offset", type=float, default=0.000)
    p.add_argument("--max-abs-y", type=float, default=0.120)

    # Quan trọng: hạ thấp ở precontact trước khi push.
    p.add_argument("--precontact-offset", type=float, default=0.070)
    p.add_argument("--contact-z", type=float, default=0.085)
    p.add_argument("--min-z", type=float, default=0.075)

    p.add_argument("--forward-step", type=float, default=0.006)
    p.add_argument("--y-step", type=float, default=0.006)
    p.add_argument("--z-step", type=float, default=0.005)

    p.add_argument("--push-step", type=float, default=0.008)
    p.add_argument("--push-repeat", type=int, default=14)
    p.add_argument("--push-y-sweep", type=float, default=0.030)
    p.add_argument("--push-y-step", type=float, default=0.004)
    p.add_argument("--push-sweep-period", type=int, default=5)
    p.add_argument("--z-hold-step", type=float, default=0.003)

    p.add_argument("--velocity", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.22)
    p.add_argument("--max-dq", type=float, default=0.090)

    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if not args.execute or args.confirm != "YES_AUTO_GREEN_PUSH":
        raise RuntimeError("Need --execute --confirm YES_AUTO_GREEN_PUSH")

    print("===== AUTO GREEN PRECONTACT PUSH =====")
    print("HSV detects green object.")
    print("Phase 1: go to precontact point while lowering Z.")
    print("Phase 2: push +X from low Z, with Y sweep and Z hold.")
    print("No model target. No backward X. No upward Z during approach.")

    print("\n===== ROBOT HOME FIRST =====")
    from pyniryo import NiryoRobot
    robot = NiryoRobot(args.ip)

    try:
        robot.set_learning_mode(False)
        robot.set_arm_max_velocity(args.velocity)
        robot.move_to_home_pose()
        time.sleep(0.8)

        print("\n===== DETECT OBJECT AFTER HOME =====")
        precontact_x, contact_x, target_y, cam_idx = capture_object_and_compute(args)

        print("\n===== CAMERA LIVE OPEN =====")
        live = R.base_live.LiveCamera(cam_idx)
        live.start()

        mode = "precontact_low"
        push_count = 0
        push_hold_y = None

        for t in range(160):
            if live.should_stop():
                print("STOP: q pressed")
                break

            joints = np.array(robot.get_joints(), dtype=np.float64)
            pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
            xyz = pose[:3]

            if mode == "precontact_low":
                # Đi tới điểm trước vật, không đi quá precontact_x trong pha hạ.
                dx = 0.0 if xyz[0] >= precontact_x else args.forward_step

                # Chỉnh Y về target_y.
                y_err = target_y - xyz[1]
                dy = clamp(y_err, -args.y_step, args.y_step)

                # Hạ Z ngay từ đầu, nhưng hạ trước khi chạm vật.
                dz = 0.0 if xyz[2] <= args.contact_z else -min(args.z_step, xyz[2] - args.contact_z)

                dxyz = np.array([dx, dy, dz], dtype=float)

                ready_x = xyz[0] >= precontact_x
                ready_y = abs(target_y - xyz[1]) <= args.y_step * 1.5
                ready_z = xyz[2] <= args.contact_z + 0.003

                if ready_x and ready_y and ready_z:
                    print("SWITCH TO PUSH")
                    mode = "push"
                    push_hold_y = float(xyz[1])
                    print("push_hold_y:", round(push_hold_y, 6))
                    continue

            else:
                # Push thẳng +X. Không quét Y nữa.
                # Khóa Y tại thời điểm bắt đầu push để tránh đang đúng lại bị kéo lệch.
                if push_hold_y is None:
                    push_hold_y = float(xyz[1])

                dy = clamp(push_hold_y - xyz[1], -0.0015, 0.0015)

                # Giữ Z quanh contact_z, tránh càng push càng tụt xuống bàn.
                dz = clamp(args.contact_z - xyz[2], -0.002, 0.002)

                dxyz = np.array([args.push_step, dy, dz], dtype=float)
                push_count += 1

                if push_count > args.push_repeat:
                    print("PUSH DONE")
                    break

            next_xyz = xyz + dxyz

            if next_xyz[2] < args.min_z:
                print("STOP: next z below min_z")
                break
            if next_xyz[0] > 0.370:
                print("STOP: x too far")
                break
            if abs(next_xyz[1]) > args.max_abs_y:
                print("STOP: y too far")
                break

            J, fk_pose = R.numerical_xyz_jacobian(robot, joints)
            dq, pred_dxyz, dq_clipped = R.solve_diffik(J, dxyz, max_dq=args.max_dq)
            target_joints = joints + dq

            print(f"\n[{t:02d}] mode={mode}")
            print("cur xyz :", np.round(xyz, 6))
            print("targets :", np.round([precontact_x, contact_x, target_y, args.contact_z], 6),
                  "[pre_x, contact_x, y, z]")
            print("cmd dxyz:", np.round(dxyz, 6))
            print("dq      :", np.round(dq, 6), "dq_clipped:", dq_clipped)
            print("wrist dq:", np.round(dq[3:6], 6))

            live.update_status([
                f"AUTO GREEN {mode.upper()} {t}",
                f"cur={np.round(xyz, 3)}",
                f"pre_x={precontact_x:.3f}",
                f"contact_x={contact_x:.3f}",
                f"target_y={target_y:.3f}",
                f"z={args.contact_z:.3f}",
                "q stop",
            ])

            before = pose.copy()
            R.move_joints_compat(robot, target_joints)
            time.sleep(args.sleep)

            new_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
            print("new xyz :", np.round(new_pose[:3], 6))
            print("actual  :", np.round(new_pose[:3] - before[:3], 6))

        final_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=float)
        print("\nfinal pose:", np.round(final_pose, 6))
        print("AUTO GREEN PRECONTACT PUSH DONE")

    finally:
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
