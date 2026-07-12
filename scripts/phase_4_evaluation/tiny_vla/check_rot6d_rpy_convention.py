#!/usr/bin/env python3
import os
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import h5py
import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
except Exception as e:
    raise RuntimeError("Need scipy. Try: pip install scipy") from e


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def matrix_from_two_columns(a1, a2):
    """
    Build rotation matrix from first two column vectors using Gram-Schmidt.
    """
    b1 = normalize(a1)
    a2 = np.asarray(a2, dtype=np.float64)
    b2 = normalize(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    M = np.stack([b1, b2, b3], axis=1)
    return M


def rpy_from_matrix(M):
    """
    For R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
    scipy as_euler('xyz') returns [roll, pitch, yaw].
    """
    return R.from_matrix(M).as_euler("xyz", degrees=False)


def convert_rot6d(r6, mode):
    r6 = np.asarray(r6, dtype=np.float64).reshape(6)

    if mode == "official_cols_3_3":
        # Common convention:
        # r6 = [col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]
        a1 = r6[0:3]
        a2 = r6[3:6]

    elif mode == "interleaved_R_col01_rowmajor":
        # Convention produced by:
        # R[:, :2].reshape(-1) in numpy row-major
        #
        # r6 = [R00, R01, R10, R11, R20, R21]
        # col1 = [R00, R10, R20] = [r0, r2, r4]
        # col2 = [R01, R11, R21] = [r1, r3, r5]
        a1 = np.array([r6[0], r6[2], r6[4]])
        a2 = np.array([r6[1], r6[3], r6[5]])

    elif mode == "rows_3_3":
        # Less likely: first 3 are row1, next 3 are row2.
        row1 = r6[0:3]
        row2 = r6[3:6]
        # Reconstruct by treating row1,row2 then deriving row3.
        b1 = normalize(row1)
        b2 = normalize(row2 - np.dot(b1, row2) * b1)
        b3 = np.cross(b1, b2)
        M_rows = np.stack([b1, b2, b3], axis=0)
        return rpy_from_matrix(M_rows)

    else:
        raise ValueError(mode)

    M = matrix_from_two_columns(a1, a2)
    return rpy_from_matrix(M)


def error_report(pred, ref):
    diff = wrap_angle(pred - ref)
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    max_abs = np.max(np.abs(diff), axis=0)
    return mae, rmse, max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.environ.get(
            "TINYVLA_DATASET_DIR",
            str(Path.home() / "TinyVLA/data/data/niryo_push_1cam_10d_50_hdf5"),
        )
    )
    ap.add_argument("--num-files", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    files = sorted(root.glob("episode_*.hdf5"))[:args.num_files]

    if not files:
        raise SystemExit(f"No HDF5 files found in {root}")

    print("===== CHECK ROT6D -> RPY CONVENTION =====")
    print("root:", root)
    print("num_files_checked:", len(files))
    print()

    modes = [
        "official_cols_3_3",
        "interleaved_R_col01_rowmajor",
        "rows_3_3",
    ]

    # collect errors
    results = {}

    for mode in modes:
        results[(mode, "target_same_index")] = []
        results[(mode, "target_next_index")] = []
        results[(mode, "ee_same_index")] = []

    example_printed = False

    for fp in files:
        with h5py.File(fp, "r") as f:
            action = np.asarray(f["action"], dtype=np.float64)
            target = np.asarray(f["observations/target_ee_pose_xyzrpy"], dtype=np.float64)
            ee = np.asarray(f["observations/ee_pose_xyzrpy"], dtype=np.float64)

            T = min(len(action), len(target), len(ee))

            for mode in modes:
                pred_rpy = np.stack([
                    convert_rot6d(action[i, 3:9], mode)
                    for i in range(T)
                ], axis=0)

                # action[i] in collector may correspond to target[i+1]
                ref_same = target[:T, 3:6]
                ref_next = target[np.minimum(np.arange(T) + 1, T - 1), 3:6]
                ref_ee = ee[:T, 3:6]

                results[(mode, "target_same_index")].append(wrap_angle(pred_rpy - ref_same))
                results[(mode, "target_next_index")].append(wrap_angle(pred_rpy - ref_next))
                results[(mode, "ee_same_index")].append(wrap_angle(pred_rpy - ref_ee))

                if not example_printed:
                    print("Example file:", fp)
                    print("action[0,3:9] rot6d:", np.round(action[0, 3:9], 6))
                    print("target[0] rpy:", np.round(target[0, 3:6], 6))
                    print("target[1] rpy:", np.round(target[1, 3:6], 6))
                    print("ee[0] rpy:", np.round(ee[0, 3:6], 6))
                    print()
                    for m in modes:
                        print(f"{m} -> rpy:", np.round(convert_rot6d(action[0, 3:9], m), 6))
                    print()
                    example_printed = True

    print("===== ERROR SUMMARY =====")
    print("Lower is better. Values are radians.")
    print("Good convention should have very small MAE/RMSE against the correct reference.")
    print()

    best = None

    for key, chunks in results.items():
        mode, refname = key
        diff = np.concatenate(chunks, axis=0)
        mae = np.mean(np.abs(diff), axis=0)
        rmse = np.sqrt(np.mean(diff ** 2, axis=0))
        max_abs = np.max(np.abs(diff), axis=0)
        score = float(np.mean(mae))

        print(f"{mode} vs {refname}")
        print("  MAE roll/pitch/yaw :", np.round(mae, 6), "mean=", round(score, 6))
        print("  RMSE roll/pitch/yaw:", np.round(rmse, 6))
        print("  MAX  roll/pitch/yaw:", np.round(max_abs, 6))
        print()

        if best is None or score < best[0]:
            best = (score, mode, refname)

    print("===== BEST MATCH =====")
    print("best_mean_MAE:", round(best[0], 8))
    print("best_mode:", best[1])
    print("best_reference:", best[2])

    print()
    print("INTERPRETATION:")
    print("- If best_mode is interleaved_R_col01_rowmajor, then the collector's R[:, :2].reshape(-1) convention is the correct one.")
    print("- If official_cols_3_3 has large error but interleaved is small, the current official conversion is mismatched.")
    print("- If all modes have large error, action rot6D may not match target_ee_pose_xyzrpy or dataset convention is different.")


if __name__ == "__main__":
    main()
