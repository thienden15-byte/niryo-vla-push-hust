import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


def collect_datasets(h5):
    out = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append({
                "name": name,
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
            })

    h5.visititems(visitor)
    return out


def find_dataset(datasets, keywords=None, ndim=None, last_dim=None):
    keywords = keywords or []

    scored = []
    for d in datasets:
        name = d["name"].lower()
        shape = d["shape"]

        score = 0
        for kw in keywords:
            if kw.lower() in name:
                score += 10

        if ndim is not None and len(shape) == ndim:
            score += 3

        if last_dim is not None and len(shape) > 0 and shape[-1] == last_dim:
            score += 3

        if score > 0:
            scored.append((score, d))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]["name"]


def find_image_dataset(datasets):
    candidates = []

    for d in datasets:
        name = d["name"].lower()
        shape = d["shape"]

        if len(shape) == 4:
            # NHWC image: T,H,W,3
            if shape[-1] == 3:
                score = 10
                if "image" in name or "camera" in name or "cam" in name:
                    score += 10
                candidates.append((score, d["name"]))

            # NCHW image: T,3,H,W
            if len(shape) == 4 and shape[1] == 3:
                score = 8
                if "image" in name or "camera" in name or "cam" in name:
                    score += 10
                candidates.append((score, d["name"]))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def read_as_numpy(h5, path):
    if path is None:
        return None
    return np.array(h5[path])


def normalize_image(img):
    arr = np.array(img)

    # Nếu ảnh dạng CHW thì đổi sang HWC
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        # Có thể ảnh nằm trong [0,1] hoặc [0,255]
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def short_stats(arr):
    if arr is None:
        return "None"

    arr = np.asarray(arr)
    if arr.size == 0:
        return f"shape={arr.shape}, empty"

    if np.issubdtype(arr.dtype, np.number):
        return (
            f"shape={arr.shape}, dtype={arr.dtype}, "
            f"min={np.nanmin(arr):.6f}, mean={np.nanmean(arr):.6f}, max={np.nanmax(arr):.6f}"
        )

    return f"shape={arr.shape}, dtype={arr.dtype}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True, help="Path tới file .hdf5/.h5 episode")
    ap.add_argument("--out", default=None, help="Ảnh report output .png")
    args = ap.parse_args()

    ep_path = Path(args.ep).expanduser()
    if not ep_path.exists():
        raise FileNotFoundError(ep_path)

    out_path = Path(args.out).expanduser() if args.out else ep_path.with_suffix(".report.png")

    with h5py.File(ep_path, "r") as h5:
        datasets = collect_datasets(h5)

        image_key = find_image_dataset(datasets)

        qpos_key = find_dataset(
            datasets,
            keywords=["qpos", "joint", "state"],
            ndim=2,
        )

        action_key = find_dataset(
            datasets,
            keywords=["action"],
            ndim=2,
        )

        instruction_key = find_dataset(
            datasets,
            keywords=["instruction", "language", "text", "prompt"],
        )

        images = read_as_numpy(h5, image_key)
        qpos = read_as_numpy(h5, qpos_key)
        actions = read_as_numpy(h5, action_key)
        instruction = read_as_numpy(h5, instruction_key)

        T = None
        for arr in [images, qpos, actions]:
            if arr is not None and arr.ndim >= 1:
                T = arr.shape[0]
                break

        if T is None:
            T = 0

        frame_ids = []
        if images is not None and images.ndim >= 4 and images.shape[0] > 0:
            n = images.shape[0]
            frame_ids = [0, n // 2, n - 1]

        lines = []
        lines.append("HDF5 EPISODE REPORT SIMPLE")
        lines.append("")
        lines.append(f"file: {ep_path}")
        lines.append(f"num datasets: {len(datasets)}")
        lines.append(f"valid_len / T guess: {T}")
        lines.append("")
        lines.append("MAIN FIELDS")
        lines.append(f"image_key      : {image_key}")
        lines.append(f"qpos/state_key : {qpos_key}")
        lines.append(f"action_key     : {action_key}")
        lines.append(f"instruction_key: {instruction_key}")
        lines.append("")
        lines.append("SHAPES / STATS")
        lines.append(f"images : {short_stats(images)}")
        lines.append(f"qpos   : {short_stats(qpos)}")
        lines.append(f"action : {short_stats(actions)}")

        if instruction is not None:
            lines.append(f"instruction raw: {instruction}")

        lines.append("")
        lines.append("ALL DATASETS")
        for d in datasets:
            lines.append(f"- {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

        print("\n".join(lines))

        # Vẽ report
        fig = plt.figure(figsize=(18, 10))

        ax_text = fig.add_axes([0.02, 0.05, 0.46, 0.90])
        ax_text.axis("off")
        ax_text.text(
            0,
            1,
            "\n".join(lines[:45]),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        if images is not None and len(frame_ids) == 3:
            for i, fid in enumerate(frame_ids):
                ax = fig.add_axes([0.52, 0.68 - i * 0.30, 0.42, 0.25])
                img = normalize_image(images[fid])
                ax.imshow(img)
                ax.set_title(f"Frame {fid}", fontsize=12)
                ax.axis("off")
        else:
            ax = fig.add_axes([0.52, 0.35, 0.42, 0.30])
            ax.axis("off")
            ax.text(0.5, 0.5, "No image dataset found", ha="center", va="center", fontsize=16)

        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print()
        print("saved report:", out_path)


if __name__ == "__main__":
    main()
