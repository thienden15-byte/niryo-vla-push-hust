#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np

try:
    from pyniryo import NiryoRobot
except Exception as e:
    print("❌ Không import được pyniryo:", e)
    raise SystemExit(1)


def interp_global(train_feat_n, train_goals, x_n, k=5, power=1.0):
    k = min(k, len(train_feat_n))

    d = np.linalg.norm(train_feat_n - x_n[None, :], axis=1)
    order = np.argsort(d)
    ids = order[:k]
    ds = d[ids]

    if ds[0] < 1e-9:
        weights = np.zeros_like(ds)
        weights[0] = 1.0
        pred = train_goals[ids[0]]
    else:
        weights = 1.0 / ((ds + 1e-6) ** power)
        weights = weights / weights.sum()
        pred = (train_goals[ids] * weights[:, None]).sum(axis=0)

    return pred.astype(np.float32), ids, ds.astype(np.float32), weights.astype(np.float32)


def predict_cluster(policy, object_feature, k=5, power=1.0):
    train_feat = policy["train_object_features"].astype(np.float32)
    train_goals = policy["train_cluster_goals"].astype(np.float32)
    train_files = np.array([str(x) for x in policy["train_episode_files"]])

    # Dùng đủ cx, cy, area. Không dùng zone vì zone trong V2 đang bị left/right/unknown.
    mean = train_feat.mean(axis=0)
    std = train_feat.std(axis=0) + 1e-6

    x = np.array(object_feature, dtype=np.float32).reshape(3,)
    x_n = (x - mean) / std
    train_feat_n = (train_feat - mean) / std

    pred90, ids, ds, weights = interp_global(
        train_feat_n=train_feat_n,
        train_goals=train_goals,
        x_n=x_n,
        k=k,
        power=power,
    )

    info = {
        "selected_files": train_files[ids],
        "selected_ds": ds,
        "selected_weights": weights,
        "cluster_idxs": policy["cluster_idxs"].astype(np.int64),
    }

    return pred90.reshape(-1, 6), info


def densify_path(start_joints, cluster_points, max_step_rad=0.020):
    waypoints = [start_joints.astype(np.float32)]
    current = start_joints.astype(np.float32)

    for target in cluster_points.astype(np.float32):
        diff = target - current
        max_abs = float(np.max(np.abs(diff)))
        steps = max(2, int(np.ceil(max_abs / max_step_rad)) + 1)

        for s in range(1, steps + 1):
            a = s / steps
            q = current * (1.0 - a) + target * a
            waypoints.append(q.astype(np.float32))

        current = target

    return np.stack(waypoints, axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument(
        "--policy",
        type=str,
        default="goal_v1/checkpoints/goal_v2_cluster_policy.npz",
    )
    parser.add_argument(
        "--object-feature",
        type=float,
        nargs=3,
        required=True,
        help="cx cy area",
    )
    parser.add_argument("--velocity", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--max-step-rad", type=float, default=0.020)
    parser.add_argument(
        "--densify",
        action="store_true",
        help="Nếu bật thì chèn thêm điểm trung gian. Mặc định tắt để gửi đúng 15 điểm AI sinh ra.",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print("❌ Không thấy policy:", policy_path)
        raise SystemExit(1)

    policy = np.load(policy_path, allow_pickle=True)

    object_feature = np.array(args.object_feature, dtype=np.float32)

    cluster_points, info = predict_cluster(
        policy,
        object_feature=object_feature,
        k=args.k,
        power=args.power,
    )

    print("=" * 100)
    print("GOAL V2 CLUSTER → NIRYO")
    print("=" * 100)
    print("Policy:", policy_path)
    print("Object feature:", object_feature)
    print("Cluster idxs:", info["cluster_idxs"].tolist())
    print("Predicted cluster points:", cluster_points.shape)
    print("\nSelected demos:")
    for f, d, w in zip(info["selected_files"], info["selected_ds"], info["selected_weights"]):
        print(f"  {str(f):12s} dist={float(d):.6f} weight={float(w):.3f}")

    print("\nPredicted 15 joint points:")
    for idx, q in zip(info["cluster_idxs"], cluster_points):
        print(f"  step {int(idx):02d}: {np.array2string(q, precision=6, suppress_small=True)}")

    if args.dry_run:
        print("\nDRY RUN: chỉ in kết quả, không chạy robot.")
        return

    print("\n⚠️ Chuẩn bị chạy ROBOT THẬT bằng PyNiryo.")
    print("Tay gần nút dừng khẩn cấp. Tốc độ đang để thấp.")
    ans = input("Gõ YES_RUN để chạy robot thật: ").strip()

    if ans != "YES_RUN":
        print("Đã hủy.")
        return

    robot = None

    try:
        print("\nKết nối robot...")
        robot = NiryoRobot(args.ip)
        robot.set_learning_mode(False)
        robot.set_arm_max_velocity(args.velocity)

        start_joints = np.array(robot.get_joints(), dtype=np.float32)
        print("Current joints:", np.array2string(start_joints, precision=6, suppress_small=True))

        if args.densify:
            path = densify_path(
                start_joints=start_joints,
                cluster_points=cluster_points,
                max_step_rad=args.max_step_rad,
            )
            print("Mode: DENSIFY - có chèn thêm điểm trung gian")
        else:
            path = cluster_points.astype(np.float32)
            print("Mode: RAW 15 POINTS - gửi đúng 15 điểm AI sinh ra")

        print("Path sent to PyNiryo:", path.shape)
        print("First:", np.array2string(path[0], precision=6, suppress_small=True))
        print("Last :", np.array2string(path[-1], precision=6, suppress_small=True))

        list_joints = [q.tolist() for q in path]

        print("\nExecuting trajectory...")
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

        print("\n✅ DONE: Goal V2 cluster đã chạy xong.")

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
