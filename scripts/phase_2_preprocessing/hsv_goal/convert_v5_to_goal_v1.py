#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def scalar(data, key, default=None):
    if key not in data.files:
        return default
    v = data[key]
    try:
        if v.shape == ():
            return v.item()
    except Exception:
        pass
    return v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset_push_real_v5")
    parser.add_argument("--out", type=str, default="goal_v1/data/push_goal_v1.npz")
    parser.add_argument("--stats", type=str, default="goal_v1/data/push_goal_v1_stats.npz")
    parser.add_argument("--steps", type=int, nargs=3, default=[25, 55, 99])
    parser.add_argument("--use-only-good", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_path = Path(args.out)
    stats_path = Path(args.stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    ep_files = sorted(dataset_dir.glob("ep_*.npz"))

    print("=" * 80)
    print("CONVERT V5 DATASET -> GOAL_V1 DATASET")
    print("=" * 80)
    print("Dataset dir:", dataset_dir)
    print("Found episodes:", len(ep_files))
    print("Goal steps:", args.steps)
    print("Use only good episodes:", args.use_only_good)
    print("=" * 80)

    if len(ep_files) == 0:
        print("❌ Không tìm thấy ep_*.npz trong", dataset_dir)
        raise SystemExit(1)

    images = []
    states = []
    goals = []
    start_joints_all = []
    object_features = []
    episode_ids = []
    episode_files = []
    robot_zones = []
    distance_bins = []
    lateral_bins = []
    success_flags = []
    object_moved_flags = []
    table_touch_flags = []

    skipped = []
    used_source_counter = Counter()
    zone_counter = Counter()
    good_counter = Counter()

    max_step = max(args.steps)

    for ep_path in ep_files:
        try:
            data = np.load(ep_path, allow_pickle=True)

            required = ["images", "joints", "start_object_feature"]
            missing = [k for k in required if k not in data.files]
            if missing:
                skipped.append((ep_path.name, f"missing {missing}"))
                continue

            valid_len = int(scalar(data, "valid_len", len(data["images"])))

            if valid_len <= max_step:
                skipped.append((ep_path.name, f"valid_len={valid_len} <= max_step={max_step}"))
                continue

            success = bool(scalar(data, "success", True))
            object_moved = bool(scalar(data, "object_moved", True))
            table_touch = bool(scalar(data, "table_touch", False))

            good = success and object_moved and (not table_touch)
            good_counter["good" if good else "bad"] += 1

            if args.use_only_good and not good:
                skipped.append((ep_path.name, f"not good: success={success}, object_moved={object_moved}, table_touch={table_touch}"))
                continue

            # Ưu tiên replay_target_joints vì đây là đường đi mục tiêu đã resample từ demo.
            # Nếu không có thì dùng joints.
            if "replay_target_joints" in data.files:
                path_joints = data["replay_target_joints"].astype(np.float32)
                source_name = "replay_target_joints"
            elif "target_joints" in data.files:
                path_joints = data["target_joints"].astype(np.float32)
                source_name = "target_joints"
            else:
                path_joints = data["joints"].astype(np.float32)
                source_name = "joints"

            used_source_counter[source_name] += 1

            actual_joints = data["joints"].astype(np.float32)
            img0 = data["images"][0].astype(np.uint8)

            start_joints = actual_joints[0].astype(np.float32)
            obj_feat = data["start_object_feature"].astype(np.float32)

            if obj_feat.shape[0] != 3:
                skipped.append((ep_path.name, f"start_object_feature shape wrong: {obj_feat.shape}"))
                continue

            # Đầu vào trạng thái: 6 góc khớp ban đầu + 3 thông tin vị trí vật
            state = np.concatenate([start_joints, obj_feat], axis=0).astype(np.float32)

            # Đầu ra: 3 tư thế khớp mục tiêu, mỗi tư thế 6 khớp -> 18 số
            goal_chunks = []
            for s in args.steps:
                goal_chunks.append(path_joints[s].astype(np.float32))
            goal = np.concatenate(goal_chunks, axis=0).astype(np.float32)

            ep_id = int(scalar(data, "episode_id", len(episode_ids)))
            zone = str(scalar(data, "robot_zone", "unknown"))
            dist_bin = str(scalar(data, "robot_distance_bin", "unknown"))
            lat_bin = str(scalar(data, "robot_lateral_bin", "unknown"))

            images.append(img0)
            states.append(state)
            goals.append(goal)
            start_joints_all.append(start_joints)
            object_features.append(obj_feat)
            episode_ids.append(ep_id)
            episode_files.append(ep_path.name)
            robot_zones.append(zone)
            distance_bins.append(dist_bin)
            lateral_bins.append(lat_bin)
            success_flags.append(success)
            object_moved_flags.append(object_moved)
            table_touch_flags.append(table_touch)

            zone_counter[zone] += 1

        except Exception as e:
            skipped.append((ep_path.name, f"exception: {e}"))

    if len(images) == 0:
        print("❌ Không có episode nào dùng được.")
        print("Skipped:")
        for item in skipped:
            print(" ", item)
        raise SystemExit(1)

    images = np.stack(images, axis=0)
    states = np.stack(states, axis=0)
    goals = np.stack(goals, axis=0)
    start_joints_all = np.stack(start_joints_all, axis=0)
    object_features = np.stack(object_features, axis=0)
    episode_ids = np.array(episode_ids, dtype=np.int32)
    episode_files = np.array(episode_files, dtype=object)
    robot_zones = np.array(robot_zones, dtype=object)
    distance_bins = np.array(distance_bins, dtype=object)
    lateral_bins = np.array(lateral_bins, dtype=object)
    success_flags = np.array(success_flags, dtype=bool)
    object_moved_flags = np.array(object_moved_flags, dtype=bool)
    table_touch_flags = np.array(table_touch_flags, dtype=bool)

    # Chia train/val theo episode
    rng = np.random.default_rng(args.seed)
    n = len(images)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * 0.2)))
    val_idx = set(indices[:n_val].tolist())

    # split = 0 là train, 1 là val
    splits = np.zeros(n, dtype=np.int32)
    for i in val_idx:
        splits[i] = 1

    # Thống kê chuẩn hóa
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-6
    goal_mean = goals.mean(axis=0)
    goal_std = goals.std(axis=0) + 1e-6

    np.savez_compressed(
        out_path,
        images=images,
        states=states,
        goals=goals,
        start_joints=start_joints_all,
        object_features=object_features,
        goal_steps=np.array(args.steps, dtype=np.int32),
        episode_ids=episode_ids,
        episode_files=episode_files,
        robot_zones=robot_zones,
        distance_bins=distance_bins,
        lateral_bins=lateral_bins,
        success=success_flags,
        object_moved=object_moved_flags,
        table_touch=table_touch_flags,
        splits=splits,
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        goal_mean=goal_mean.astype(np.float32),
        goal_std=goal_std.astype(np.float32),
        description=np.array("goal_v1: image0 + 6 start joints + 3 object feature -> 3 joint goals at selected steps", dtype=object),
    )

    np.savez_compressed(
        stats_path,
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        goal_mean=goal_mean.astype(np.float32),
        goal_std=goal_std.astype(np.float32),
        goal_steps=np.array(args.steps, dtype=np.int32),
    )

    print("\n✅ DONE")
    print("Saved dataset:", out_path)
    print("Saved stats  :", stats_path)

    print("\nShapes:")
    print("  images          :", images.shape, images.dtype)
    print("  states          :", states.shape, states.dtype, "  # 6 joints + 3 object feature")
    print("  goals           :", goals.shape, goals.dtype, "  # 3 poses x 6 joints = 18")
    print("  object_features :", object_features.shape, object_features.dtype)
    print("  splits          :", splits.shape, "train:", int((splits == 0).sum()), "val:", int((splits == 1).sum()))

    print("\nGoal steps:")
    print("  approach-ish :", args.steps[0])
    print("  contact-ish  :", args.steps[1])
    print("  end-push     :", args.steps[2])

    print("\nJoint source used:")
    for k, v in used_source_counter.items():
        print(f"  {k}: {v}")

    print("\nGood/bad raw count:")
    for k, v in good_counter.items():
        print(f"  {k}: {v}")

    print("\nZone count used:")
    for k in sorted(zone_counter.keys()):
        print(f"  {k:15s}: {zone_counter[k]}")

    print("\nSkipped:", len(skipped))
    if skipped:
        for name, reason in skipped[:20]:
            print(" ", name, "->", reason)
        if len(skipped) > 20:
            print("  ...")

    print("=" * 80)

if __name__ == "__main__":
    main()
