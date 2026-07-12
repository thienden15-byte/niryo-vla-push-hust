import os, sys, time, argparse, importlib.util
from pathlib import Path
import numpy as np

TINYVLA_REPO = Path(
    os.environ.get("TINYVLA_REPO", str(Path.home() / "TinyVLA"))
).expanduser()

sys.path.insert(0, str(TINYVLA_REPO))
sys.path.insert(0, str(TINYVLA_REPO / "llava-pythia"))

base_live_path = Path(__file__).resolve().parents[3] / "scripts/common/tiny_vla/run_author10d_fixed50_xyz_chunk_live.py"
spec = importlib.util.spec_from_file_location("base_live", base_live_path)
base_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_live)


def pose_to_list(pose):
    if hasattr(pose, "to_list"):
        return list(pose.to_list()[:6])
    if isinstance(pose, (list, tuple, np.ndarray)):
        return list(pose[:6])
    return [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]


def call_fk(robot, joints):
    joints = np.array(joints, dtype=float)
    try:
        return np.array(pose_to_list(robot.forward_kinematics(joints.tolist())), dtype=float)
    except Exception:
        return np.array(pose_to_list(robot.forward_kinematics(*joints.tolist())), dtype=float)


def numerical_xyz_jacobian(robot, joints, eps=1e-2):
    joints = np.array(joints, dtype=float)
    base_xyz = call_fk(robot, joints)[:3]
    J = np.zeros((3, 6), dtype=float)

    for i in range(6):
        jp = joints.copy()
        jp[i] += eps
        pp = call_fk(robot, jp)
        J[:, i] = (pp[:3] - base_xyz) / eps

    return J


def solve_diffik(J, dxyz, max_dq=0.05, damping=0.03):
    W = np.diag([1.0, 1.0, 1.2, 5.0, 5.0, 5.0])
    A = np.vstack([J, damping * W])
    b = np.concatenate([dxyz, np.zeros(6)])
    dq, *_ = np.linalg.lstsq(A, b, rcond=None)

    m = float(np.max(np.abs(dq)))
    if m > max_dq:
        dq = dq / m * max_dq

    return dq


def move_joints_compat(robot, joints):
    try:
        robot.move_joints(*joints.tolist())
    except TypeError:
        robot.move_joints(joints.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--dz", type=float, default=-0.002)
    p.add_argument("--min-z", type=float, default=0.102)
    p.add_argument("--velocity", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.35)
    p.add_argument("--max-dq", type=float, default=0.05)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if not args.execute or args.confirm != "YES_CONTACT_DESCENT":
        raise RuntimeError("Need --execute --confirm YES_CONTACT_DESCENT")

    print("===== CONTACT DESCENT ASSIST =====")
    print("dz per step:", args.dz)
    print("steps:", args.steps)
    print("min_z:", args.min_z)

    print("\n===== CAMERA OPEN =====")
    cam_idx, cam_name, candidates = base_live.find_usb_camera()
    print("camera candidates:", candidates)
    print("auto selected:", cam_idx, cam_name)

    live = base_live.LiveCamera(cam_idx)
    live.start()

    from pyniryo import NiryoRobot
    robot = NiryoRobot(args.ip)
    robot.set_learning_mode(False)
    robot.set_arm_max_velocity(args.velocity)

    try:
        for i in range(args.steps):
            if live.should_stop():
                print("STOP: q pressed")
                break

            joints = np.array(robot.get_joints(), dtype=float)
            pose = np.array(pose_to_list(robot.get_pose()), dtype=float)
            xyz = pose[:3]

            print(f"\n===== DESCENT STEP {i+1}/{args.steps} =====")
            print("current xyz:", np.round(xyz, 6))

            if xyz[2] + args.dz < args.min_z:
                print("STOP: would go below min_z")
                break

            dxyz = np.array([0.0, 0.0, args.dz], dtype=float)

            J = numerical_xyz_jacobian(robot, joints)
            dq = solve_diffik(J, dxyz, max_dq=args.max_dq)

            target_joints = joints + dq
            pred_pose = call_fk(robot, target_joints)

            print("desired dxyz:", np.round(dxyz, 6))
            print("dq:", np.round(dq, 6))
            print("pred xyz:", np.round(pred_pose[:3], 6))
            print("pred dxyz:", np.round(pred_pose[:3] - xyz, 6))
            print("wrist dq j4/j5/j6:", np.round(dq[3:6], 6))

            live.update_status([
                f"CONTACT DESCENT {i+1}/{args.steps}",
                f"cur z={xyz[2]:.3f}",
                f"dz={args.dz:.3f}",
                "press q to stop"
            ])

            move_joints_compat(robot, target_joints)
            time.sleep(args.sleep)

            new_pose = np.array(pose_to_list(robot.get_pose()), dtype=float)
            print("new xyz:", np.round(new_pose[:3], 6))
            print("actual dxyz:", np.round(new_pose[:3] - xyz, 6))

        print("\nCONTACT DESCENT DONE")

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
