import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def collect_datasets(h5):
    out = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, tuple(obj.shape), str(obj.dtype)))

    h5.visititems(visitor)
    return out


def find_image_key(datasets):
    cands = []
    for name, shape, dtype in datasets:
        lname = name.lower()
        if len(shape) == 4 and (shape[-1] == 3 or shape[1] == 3):
            score = 1
            if "image" in lname or "camera" in lname or "cam" in lname:
                score += 10
            cands.append((score, name))
    if not cands:
        return None
    return sorted(cands, reverse=True)[0][1]


def find_key(datasets, keywords, ndim=None):
    cands = []
    for name, shape, dtype in datasets:
        lname = name.lower()
        score = 0
        for kw in keywords:
            if kw in lname:
                score += 10
        if ndim is not None and len(shape) == ndim:
            score += 2
        if score > 0:
            cands.append((score, name))
    if not cands:
        return None
    return sorted(cands, reverse=True)[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-csv", default="hdf5_dataset_summary.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))

    print("root:", root)
    print("num hdf5 files:", len(files))

    rows = []

    for i, f in enumerate(files):
        try:
            with h5py.File(f, "r") as h5:
                datasets = collect_datasets(h5)

                image_key = find_image_key(datasets)
                qpos_key = find_key(datasets, ["qpos", "joint", "state"], ndim=2)
                action_key = find_key(datasets, ["action"], ndim=2)

                image_shape = tuple(h5[image_key].shape) if image_key else None
                qpos_shape = tuple(h5[qpos_key].shape) if qpos_key else None
                action_shape = tuple(h5[action_key].shape) if action_key else None

                T = None
                for shape in [image_shape, qpos_shape, action_shape]:
                    if shape is not None and len(shape) >= 1:
                        T = shape[0]
                        break

                qpos_dim = qpos_shape[-1] if qpos_shape is not None and len(qpos_shape) >= 2 else None
                action_dim = action_shape[-1] if action_shape is not None and len(action_shape) >= 2 else None

                rows.append({
                    "idx": i,
                    "file": str(f),
                    "T": T,
                    "image_key": image_key,
                    "image_shape": str(image_shape),
                    "qpos_key": qpos_key,
                    "qpos_shape": str(qpos_shape),
                    "qpos_dim": qpos_dim,
                    "action_key": action_key,
                    "action_shape": str(action_shape),
                    "action_dim": action_dim,
                    "num_datasets": len(datasets),
                    "status": "OK",
                })

        except Exception as e:
            rows.append({
                "idx": i,
                "file": str(f),
                "status": f"ERROR: {e}",
            })

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv).expanduser()
    df.to_csv(out_csv, index=False)

    print()
    print("===== SUMMARY =====")
    if len(df) > 0 and "T" in df:
        print("episodes:", len(df))
        print("T min/mean/max:", df["T"].min(), df["T"].mean(), df["T"].max())
        print("qpos_dim counts:")
        print(df["qpos_dim"].value_counts(dropna=False))
        print("action_dim counts:")
        print(df["action_dim"].value_counts(dropna=False))
        print()
        print(df[["idx", "T", "qpos_dim", "action_dim", "image_shape", "qpos_shape", "action_shape", "status"]].head(20).to_string(index=False))

    print()
    print("saved csv:", out_csv)


if __name__ == "__main__":
    main()
