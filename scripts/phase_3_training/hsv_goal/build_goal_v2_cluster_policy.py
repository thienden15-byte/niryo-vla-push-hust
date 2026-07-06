#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter

import numpy as np


CLUSTER_IDXS = np.array(
    [15, 18, 20, 22, 25,
     55, 58, 60, 62, 65,
     80, 85, 90, 95, 99],
    dtype=np.int64,
)


def get_first_existing(data, keys):
    for k in keys:
        if k in data:
            return data[k]
    return None


def as_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return as_str(x.item())
        if len(x) > 0:
            return as_str(x[0])
    return str(x)


def majority_vote(labels):
    labels = [str(x) for x in labels]
    c = Counter(labels)
    max_count = max(c.values())
    candidates = sorted([k for k, v in c.items() if v == max_count])
    return candidates[0]


def interp(train_feat_pool, goal_pool, x, k=5, power=1.0):
    k = min(k, len(train_feat_pool))
    d = np.linalg.norm(train_feat_pool - x[None, :], axis=1)
    order = np.argsort(d)
    ids = order[:k]
    ds = d[ids]

    if ds[0] < 1e-9:
        weights = np.zeros_like(ds)
        weights[0] = 1.0
        pred = goal_pool[ids[0]]
    else:
        weights = 1.0 / ((ds + 1e-6) ** power)
        weights = weights / weights.sum()
        pred = (goal_pool[ids] * weights[:, None]).sum(axis=0)

    return pred.astype(np.float32), ids, ds.astype(np.float32), weights.astype(np.float32)


def load_episodes(dataset_dir):
    files = sorted(Path(dataset_dir).glob("ep_*.npz"))

    if not files:
        raise FileNotFoundError(f"Không thấy ep_*.npz trong {dataset_dir}")

    episode_files = []
    object_features = []
    zones = []
    cluster_goals = []

    skipped = []

    for fp in files:
        data = np.load(fp, allow_pickle=True)

        joints = get_first_existing(data, ["joints", "actual_joints"])
        if joints is None:
            skipped.append((fp.name, "missing joints"))
            continue

        joints = np.asarray(joints, dtype=np.float32)

        valid_len = int(data["valid_len"]) if "valid_len" in data else len(joints)
        joints = joints[:valid_len]

        if len(joints) <= int(CLUSTER_IDXS.max()):
            skipped.append((fp.name, f"too short len={len(joints)}"))
            continue

        feat = get_first_existing(
            data,
            ["start_object_feature", "object_feature", "object_features"]
        )

        if feat is None:
            skipped.append((fp.name, "missing object_feature"))
            continue

        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if feat.shape[0] < 3:
            skipped.append((fp.name, f"bad object_feature shape {feat.shape}"))
            continue

        feat = feat[:3]

        zone = get_first_existing(
            data,
            ["start_zone", "robot_zone", "zone"]
        )
        if zone is None:
            zone = "unknown"
        zone = as_str(zone)

        goal = joints[CLUSTER_IDXS].reshape(-1).astype(np.float32)

        episode_files.append(fp.name)
        object_features.append(feat)
        zones.append(zone)
        cluster_goals.append(goal)

    if not episode_files:
        raise RuntimeError("Không load được episode nào.")

    return {
        "episode_files": np.array(episode_files),
        "object_features": np.stack(object_features).astype(np.float32),
        "zones": np.array(zones),
        "cluster_goals": np.stack(cluster_goals).astype(np.float32),
        "skipped": skipped,
    }


def make_split(episode_files, v1_policy_path=None, val_ratio=0.2, seed=42):
    n = len(episode_files)
    splits = np.array(["train"] * n, dtype=object)

    if v1_policy_path and Path(v1_policy_path).exists():
        p = np.load(v1_policy_path, allow_pickle=True)

        if "train_episode_files" in p and "val_episode_files" in p:
            train_set = set(str(x) for x in p["train_episode_files"])
            val_set = set(str(x) for x in p["val_episode_files"])

            for i, f in enumerate(episode_files):
                if str(f) in val_set:
                    splits[i] = "val"
                elif str(f) in train_set:
                    splits[i] = "train"

            if np.sum(splits == "val") > 0:
                return splits

    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    rng.shuffle(ids)

    n_val = max(1, int(round(n * val_ratio)))
    val_ids = ids[:n_val]
    splits[val_ids] = "val"

    return splits


def evaluate_policy(object_features, goals, zones, splits):
    train_mask = splits == "train"
    val_mask = splits == "val"

    train_feat = object_features[train_mask]
    val_feat = object_features[val_mask]

    train_goals = goals[train_mask]
    val_goals = goals[val_mask]

    train_zones = zones[train_mask]
    val_zones = zones[val_mask]

    # Dùng cùng kiểu tốt nhất V1: classifier zone bằng cy_area
    train_cy_area = train_feat[:, [1, 2]]
    val_cy_area = val_feat[:, [1, 2]]

    mean = train_cy_area.mean(axis=0)
    std = train_cy_area.std(axis=0) + 1e-6

    train_cy_area_n = (train_cy_area - mean) / std
    val_cy_area_n = (val_cy_area - mean) / std

    classifier_k = 2
    interp_k = 5
    power = 1.0

    preds = []
    pred_zones = []
    selected_info = []

    for x in val_cy_area_n:
        d = np.linalg.norm(train_cy_area_n - x[None, :], axis=1)
        order = np.argsort(d)

        cls_ids = order[:classifier_k]
        cls_labels = train_zones[cls_ids]
        pred_zone = majority_vote(cls_labels)

        pool_pos = np.where(train_zones == pred_zone)[0]
        if len(pool_pos) == 0:
            pool_pos = np.arange(len(train_goals))

        pred, ids_in_pool, ds, weights = interp(
            train_feat_pool=train_cy_area_n[pool_pos],
            goal_pool=train_goals[pool_pos],
            x=x,
            k=interp_k,
            power=power,
        )

        selected_pos = pool_pos[ids_in_pool]

        preds.append(pred)
        pred_zones.append(pred_zone)
        selected_info.append((selected_pos, ds, weights))

    preds = np.stack(preds)

    abs_err = np.abs(preds - val_goals)
    mae_all = float(abs_err.mean())

    # MAE theo 15 điểm, mỗi điểm 6 joint
    err_by_point = abs_err.reshape(len(val_goals), len(CLUSTER_IDXS), 6).mean(axis=(0, 2))

    zone_acc = float(np.mean(np.array(pred_zones) == val_zones)) if len(val_zones) else 0.0

    result = {
        "mae_all": mae_all,
        "err_by_point": err_by_point,
        "zone_acc": zone_acc,
        "classifier_k": classifier_k,
        "interp_k": interp_k,
        "power": power,
        "cy_area_mean": mean.astype(np.float32),
        "cy_area_std": std.astype(np.float32),
        "train_cy_area_n": train_cy_area_n.astype(np.float32),
        "val_cy_area_n": val_cy_area_n.astype(np.float32),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_push_real_v5",
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default="goal_v1/data/push_goal_v2_cluster.npz",
    )
    parser.add_argument(
        "--out-policy",
        type=str,
        default="goal_v1/checkpoints/goal_v2_cluster_policy.npz",
    )
    parser.add_argument(
        "--v1-policy",
        type=str,
        default="goal_v1/checkpoints/goal_v1_retrieval_policy.npz",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("BUILD GOAL V2 CLUSTER POLICY")
    print("=" * 100)
    print("Dataset dir:", args.dataset_dir)
    print("Cluster idx:", CLUSTER_IDXS.tolist())

    loaded = load_episodes(args.dataset_dir)

    episode_files = loaded["episode_files"]
    object_features = loaded["object_features"]
    zones = loaded["zones"]
    cluster_goals = loaded["cluster_goals"]

    splits = make_split(
        episode_files,
        v1_policy_path=args.v1_policy,
    )

    print("Loaded episodes:", len(episode_files))
    print("Skipped:", len(loaded["skipped"]))

    if loaded["skipped"]:
        print("Skipped detail:")
        for name, reason in loaded["skipped"][:10]:
            print(" ", name, reason)

    print("object_features:", object_features.shape)
    print("cluster_goals :", cluster_goals.shape)
    print("train:", int(np.sum(splits == "train")))
    print("val  :", int(np.sum(splits == "val")))

    print("\nZones:")
    for z in sorted(set(zones)):
        print(f"  {z:14s}: {int(np.sum(zones == z))}")

    Path(args.out_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_policy).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out_data,
        episode_files=episode_files,
        object_features=object_features,
        zones=zones,
        cluster_goals=cluster_goals,
        splits=splits,
        cluster_idxs=CLUSTER_IDXS,
    )

    eval_result = evaluate_policy(
        object_features=object_features,
        goals=cluster_goals,
        zones=zones,
        splits=splits,
    )

    print("\nValidation result:")
    print("  MAE all :", eval_result["mae_all"])
    print("  zone acc:", eval_result["zone_acc"])
    print("  MAE by cluster point:")
    for idx, e in zip(CLUSTER_IDXS, eval_result["err_by_point"]):
        print(f"    step {int(idx):02d}: {float(e):.6f}")

    train_mask = splits == "train"
    val_mask = splits == "val"

    train_feat = object_features[train_mask]
    train_goals = cluster_goals[train_mask]
    train_zones = zones[train_mask]
    train_files = episode_files[train_mask]

    val_feat = object_features[val_mask]
    val_goals = cluster_goals[val_mask]
    val_zones = zones[val_mask]
    val_files = episode_files[val_mask]

    np.savez_compressed(
        args.out_policy,
        policy_version="goal_v2_cluster_retrieval",
        cluster_idxs=CLUSTER_IDXS,
        train_object_features=train_feat.astype(np.float32),
        train_cluster_goals=train_goals.astype(np.float32),
        train_robot_zones=train_zones,
        train_episode_files=train_files,
        val_object_features=val_feat.astype(np.float32),
        val_cluster_goals=val_goals.astype(np.float32),
        val_robot_zones=val_zones,
        val_episode_files=val_files,
        train_cy_area_n=eval_result["train_cy_area_n"],
        cy_area_mean=eval_result["cy_area_mean"],
        cy_area_std=eval_result["cy_area_std"],
        classifier_k=eval_result["classifier_k"],
        zone_interp_k=eval_result["interp_k"],
        interp_power=eval_result["power"],
        val_mae_all=eval_result["mae_all"],
        val_zone_acc=eval_result["zone_acc"],
        val_mae_by_point=eval_result["err_by_point"],
    )

    print("\nSaved:")
    print(" ", args.out_data)
    print(" ", args.out_policy)
    print("=" * 100)


if __name__ == "__main__":
    main()
