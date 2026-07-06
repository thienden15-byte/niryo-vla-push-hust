#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, glob, time, math
from pathlib import Path

import cv2
import h5py
import numpy as np
from pyniryo import NiryoRobot


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def next_ep_path(out_dir):
    ensure_dir(out_dir)
    files = sorted(glob.glob(os.path.join(out_dir, "episode_*.hdf5")))
    mx = -1
    for f in files:
        try:
            mx = max(mx, int(Path(f).stem.split("_")[-1]))
        except Exception:
            pass
    return os.path.join(out_dir, f"episode_{mx + 1}.hdf5")


def get_joints6(robot):
    j = robot.get_joints()
    if hasattr(j, "to_list"):
        j = j.to_list()
    return np.array(list(j)[:6], dtype=np.float32)


def get_pose6(robot):
    p = robot.get_pose()

    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    if all(hasattr(p, n) for n in names):
        return np.array([float(getattr(p, n)) for n in names], dtype=np.float32)

    if hasattr(p, "to_list"):
        v = p.to_list()
        return np.array(v[:6], dtype=np.float32)

    v = list(p)
    return np.array(v[:6], dtype=np.float32)


def set_learning(robot, enabled):
    try:
        robot.set_learning_mode(bool(enabled))
        print(f"[robot] learning_mode={enabled}")
    except Exception as e:
        print("[warn] set_learning_mode failed:", e)


def set_velocity(robot, v):
    try:
        robot.set_arm_max_velocity(int(v))
        print(f"[robot] velocity={v}")
    except Exception as e:
        print("[warn] set_arm_max_velocity failed:", e)


def move_joints(robot, j6):
    j6 = [float(x) for x in np.array(j6).reshape(-1)[:6]]
    try:
        robot.move_joints(j6)
    except Exception:
        robot.move_joints(*j6)


def resample(seq, n):
    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) == 0:
        raise RuntimeError("Empty trajectory")
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def rpy_to_rotmat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float32)
    ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float32)
    rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float32)
    return rz @ ry @ rx


def pose6_to_action10(pose6, gripper):
    x, y, z, r, p, yw = [float(v) for v in pose6]
    R = rpy_to_rotmat(r, p, yw)
    rot6d = R[:, :2].reshape(-1).astype(np.float32)
    return np.concatenate([
        np.array([x, y, z], dtype=np.float32),
        rot6d,
        np.array([float(gripper)], dtype=np.float32)
    ]).astype(np.float32)


def save_hdf5(path, images, qpos6, ee_pose6, target_pose6, args):
    images = np.asarray(images, dtype=np.uint8)
    qpos6 = np.asarray(qpos6, dtype=np.float32)
    ee_pose6 = np.asarray(ee_pose6, dtype=np.float32)
    target_pose6 = np.asarray(target_pose6, dtype=np.float32)

    T = args.episode_len

    grip = np.full((T, 1), float(args.gripper_state), dtype=np.float32)
    qpos7 = np.concatenate([qpos6, grip], axis=1).astype(np.float32)

    qvel7 = np.zeros_like(qpos7)
    if T > 1:
        qvel7[1:] = (qpos7[1:] - qpos7[:-1]) * float(args.replay_hz)
        qvel7[0] = qvel7[1]

    next_pose = np.zeros_like(target_pose6)
    next_pose[:-1] = target_pose6[1:]
    next_pose[-1] = target_pose6[-1]

    action10 = np.stack(
        [pose6_to_action10(p, args.gripper_state) for p in next_pose],
        axis=0
    ).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["compress"] = False
        f.attrs["camera_names"] = args.cam_name
        f.attrs["instruction"] = args.instruction
        f.attrs["collector"] = "niryo_1cam_10d_push_only_then_return"
        f.attrs["return_motion_recorded"] = False
        f.attrs["action_format"] = "absolute_ee_pose_10d_xyz_rot6d_gripper"

        f.create_dataset("action", data=action10, dtype="float32")

        # Dạng này đúng hơn scalar: root['language_raw'][0].decode(...)
        f.create_dataset(
            "language_raw",
            data=np.array([args.instruction.encode("utf-8")], dtype="S256")
        )

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos7, dtype="float32")
        obs.create_dataset("joint_positions", data=qpos7, dtype="float32")
        obs.create_dataset("qvel", data=qvel7, dtype="float32")
        obs.create_dataset("ee_pose_xyzrpy", data=ee_pose6, dtype="float32")
        obs.create_dataset("target_ee_pose_xyzrpy", data=target_pose6, dtype="float32")

        img_g = obs.create_group("images")
        img_g.create_dataset(
            args.cam_name,
            data=images,
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
            chunks=(1, images.shape[1], images.shape[2], images.shape[3])
        )

    print("[saved]", path)
    print(" image :", images.shape)
    print(" qpos  :", qpos7.shape)
    print(" action:", action10.shape)


def countdown(sec):
    for i in range(int(sec), 0, -1):
        print("[countdown]", i)
        time.sleep(1)


def replay_collect_save_return(robot, cap, taught_joints, taught_poses, args):
    if len(taught_joints) < 2:
        print("[error] No taught trajectory")
        return

    target_joints = resample(taught_joints, args.episode_len)
    target_poses = resample(taught_poses, args.episode_len)
    start_joints = target_joints[0].copy()

    set_learning(robot, False)
    set_velocity(robot, args.velocity)

    print("[start] move to first point WITHOUT recording")
    move_joints(robot, start_joints)
    time.sleep(args.start_settle)

    print("[collect] remove hand from camera")
    countdown(args.countdown)

    images, qpos6, ee_pose6, target_pose6 = [], [], [], []
    period = 1.0 / float(args.replay_hz)

    for i in range(args.episode_len):
        t0 = time.time()

        move_joints(robot, target_joints[i])

        if args.after_move_sleep > 0:
            time.sleep(args.after_move_sleep)

        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")

        frame_bgr = cv2.resize(frame_bgr, (args.width, args.height))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        images.append(frame_rgb)
        qpos6.append(get_joints6(robot))
        ee_pose6.append(get_pose6(robot))
        target_pose6.append(target_poses[i])

        print(f"[collect] {i+1:03d}/{args.episode_len}")

        dt = time.time() - t0
        if period - dt > 0:
            time.sleep(period - dt)

    out_path = next_ep_path(args.out_dir)
    save_hdf5(out_path, images, qpos6, ee_pose6, target_pose6, args)

    if not args.no_auto_return:
        print("[return] return to start WITHOUT recording")
        set_velocity(robot, args.return_velocity)
        move_joints(robot, start_joints)
        time.sleep(args.start_settle)
        print("[return] done")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--robot-ip", default="169.254.200.200")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--cam-name", default="front")
    ap.add_argument("--out-dir", default="data/niryo_push_1cam_10d_50_hdf5")
    ap.add_argument("--instruction", default="push the green object to the right")

    ap.add_argument("--episode-len", type=int, default=50)
    ap.add_argument("--teach-hz", type=float, default=10.0)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--replay-hz", type=float, default=8.0)

    ap.add_argument("--velocity", type=int, default=30)
    ap.add_argument("--return-velocity", type=int, default=30)

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--start-settle", type=float, default=0.3)
    ap.add_argument("--after-move-sleep", type=float, default=0.0)
    ap.add_argument("--gripper-state", type=float, default=0.0)
    ap.add_argument("--no-auto-return", action="store_true")

    args = ap.parse_args()
    args.teach_hz = args.sample_hz if args.sample_hz else args.teach_hz

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("[robot] connecting", args.robot_ip)
    robot = NiryoRobot(args.robot_ip)
    print("[robot] connected")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    mode = "READY"
    active_joints, active_poses = [], []
    taught_joints, taught_poses = [], []
    last_t = 0.0

    print("")
    print("=== Niryo TinyVLA 1cam 10D collector ===")
    print("s = teach by hand, no images")
    print("p = save taught trajectory")
    print("e = replay push and record, then return without recording")
    print("h = return to start without recording")
    print("x = discard trajectory")
    print("q = quit")
    print("================================ start without recording")
    print("x =========")
    print("")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            now = time.time()

            if mode == "TEACH" and now - last_t >= 1.0 / args.teach_hz:
                last_t = now
                try:
                    active_joints.append(get_joints6(robot))
                    active_poses.append(get_pose6(robot))
                except Exception as e:
                    print("[teach warn]", e)

            show = frame.copy()
            color = (0, 0, 255) if mode == "TEACH" else (0, 255, 0)

            cv2.putText(
                show,
                f"MODE={mode} teach={len(active_joints)} saved={len(taught_joints)} ep_len={args.episode_len}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.putText(
                show,
                "s=teach p=save e=record h=home x=discard q=quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.imshow("Niryo 1cam 10D collector", show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                mode = "TEACH"
                active_joints, active_poses = [], []
                last_t = 0.0
                set_learning(robot, True)
                print("[teach] START: approach -> push -> stop. Do NOT teach return.")

            elif key == ord("p"):
                if mode != "TEACH":
                    print("[warn] press s first")
                    continue

                if len(active_joints) < 2:
                    print("[warn] too few teach samples")
                    set_learning(robot, False)
                    mode = "READY"
                    continue

                taught_joints = [x.copy() for x in active_joints]
                taught_poses = [x.copy() for x in active_poses]
                active_joints, active_poses = [], []
                set_learning(robot, False)
                mode = "READY"

                print("[teach] SAVED samples:", len(taught_joints))
                print("[teach] reset object, remove hand, press e")

            elif key == ord("e"):
                if len(taught_joints) < 2:
                    print("[warn] no trajectory. press s -> p first")
                    continue

                mode = "COLLECT"
                try:
                    replay_collect_save_return(robot, cap, taught_joints, taught_poses, args)
                except Exception as e:
                    print("[error]", e)
                mode = "READY"

            elif key == ord("h"):
                if len(taught_joints) < 2:
                    print("[warn] no saved trajectory")
                    continue
                set_learning(robot, False)
                set_velocity(robot, args.return_velocity)
                move_joints(robot, taught_joints[0])
                print("[home] returned without recording")

            elif key == ord("x"):
                taught_joints, taught_poses = [], []
                active_joints, active_poses = [], []
                mode = "READY"
                print("[reset] discarded trajectory")

    finally:
        try:
            set_learning(robot, False)
        except Exception:
            pass
        try:
            robot.close_connection()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("[exit]")


if __name__ == "__main__":
    main()
