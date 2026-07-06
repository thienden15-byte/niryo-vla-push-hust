import os, sys, time, argparse, importlib.util
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.expanduser("~/TinyVLA"))
sys.path.insert(0, os.path.expanduser("~/TinyVLA/llava-pythia"))

diffik_path = Path.home() / "tinyvla_niryo_runtime/scripts/run_author_style_niryo_diffik_rollout.py"
spec = importlib.util.spec_from_file_location("diffik_runtime", diffik_path)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")
    p.add_argument("--approach-x", type=float, default=0.300)
    p.add_argument("--contact-z", type=float, default=0.085)
    p.add_argument("--min-z", type=float, default=0.075)
    p.add_argument("--forward-step", type=float, default=0.006)
    p.add_argument("--z-step", type=float, default=0.005)
    p.add_argument("--push-step", type=float, default=0.008)
    p.add_argument("--push-repeat", type=int, default=10)
    p.add_argument("--velocity", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--max-dq", type=float, default=0.070)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if not args.execute or args.confirm != "YES_MONOTONIC_PUSH":
        raise RuntimeError("Need --execute --confirm YES_MONOTONIC_PUSH")

    print("===== MONOTONIC APPROACH + DESCEND + PUSH =====")
    print("NO model target in motion.")
    print("NO backward X during approach.")
    print("NO upward Z during approach.")
    print("approach_x:", args.approach_x)
    print("contact_z :", args.contact_z)

    print("\n===== CAMERA OPEN =====")
    cam_idx, cam_name, candidates = R.base_live.find_usb_camera()
    print("camera candidates:", candidates)
    print("auto selected:", cam_idx, cam_name)

    live = R.base_live.LiveCamera(cam_idx)
    live.start()

    from pyniryo import NiryoRobot
    robot = NiryoRobot(args.ip)

    try:
        robot.set_learning_mode(False)
        robot.set_arm_max_velocity(args.velocity)

        print("\n===== MOVE HOME =====")
        robot.move_to_home_pose()
        time.sleep(0.8)

        mode = "approach_descend"
        push_count = 0

        for t in range(90):
            if live.should_stop():
                print("STOP: q pressed")
                break

            joints = np.array(robot.get_joints(), dtype=np.float64)
            pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
            xyz = pose[:3]

            if mode == "approach_descend":
                dx = 0.0 if xyz[0] >= args.approach_x else args.forward_step
                dz = 0.0 if xyz[2] <= args.contact_z else -min(args.z_step, xyz[2] - args.contact_z)

                dxyz = np.array([dx, 0.0, dz], dtype=float)

                if xyz[0] >= args.approach_x and xyz[2] <= args.contact_z + 0.003:
                    print("SWITCH TO PUSH")
                    mode = "push"
                    continue

            else:
                dxyz = np.array([args.push_step, 0.0, 0.0], dtype=float)
                push_count += 1
                if push_count > args.push_repeat:
                    print("PUSH DONE")
                    break

            next_xyz = xyz + dxyz

            if next_xyz[2] < args.min_z:
                print("STOP: next z below min_z")
                break
            if next_xyz[0] > 0.340:
                print("STOP: x too far")
                break

            J, fk_pose = R.numerical_xyz_jacobian(robot, joints)
            dq, pred_dxyz, dq_clipped = R.solve_diffik(J, dxyz, max_dq=args.max_dq)

            target_joints = joints + dq

            print(f"\n[{t:02d}] mode={mode}")
            print("cur xyz :", np.round(xyz, 6))
            print("cmd dxyz:", np.round(dxyz, 6))
            print("dq      :", np.round(dq, 6), "dq_clipped:", dq_clipped)
            print("wrist dq:", np.round(dq[3:6], 6))

            live.update_status([
                f"{mode.upper()} {t}",
                f"cur={np.round(xyz, 3)}",
                f"cmd={np.round(dxyz, 3)}",
                f"contact_z={args.contact_z}",
                "q stop"
            ])

            before = pose.copy()

            R.move_joints_compat(robot, target_joints)
            time.sleep(args.sleep)

            new_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
            print("new xyz :", np.round(new_pose[:3], 6))
            print("actual  :", np.round(new_pose[:3] - before[:3], 6))

        final_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=float)
        print("\nfinal pose:", np.round(final_pose, 6))
        print("MONOTONIC PUSH DONE")

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
