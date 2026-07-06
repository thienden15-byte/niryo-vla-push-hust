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


def find_dataset(datasets, keywords=None, ndim=None):
    keywords = keywords or []
    candidates = []

    for d in datasets:
        name = d["name"].lower()
        shape = d["shape"]
        score = 0

        for kw in keywords:
            if kw.lower() in name:
                score += 10

        if ndim is not None and len(shape) == ndim:
            score += 3

        if score > 0:
            candidates.append((score, d["name"]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_image_dataset(datasets):
    candidates = []

    for d in datasets:
        name = d["name"].lower()
        shape = d["shape"]
        score = 0

        if len(shape) == 4:
            # NHWC
            if shape[-1] == 3:
                score += 10
            # NCHW
            elif shape[1] == 3:
                score += 8

            if "image" in name or "camera" in name or "cam" in name or "rgb" in name:
                score += 10

        if score > 0:
            candidates.append((score, d["name"]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def normalize_image(img):
    arr = np.array(img)

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def stat_line(arr):
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


def short_text(x, max_len=90):
    s = str(x)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True, help="Path tới file .hdf5/.h5")
    ap.add_argument("--out-prefix", required=True, help="Prefix output, ví dụ ~/TinyVLA/report_ep021")
    ap.add_argument("--num-frames", type=int, default=5, choices=[3, 5], help="Số frame hiển thị")
    args = ap.parse_args()

    ep_path = Path(args.ep).expanduser()
    out_prefix = Path(args.out_prefix).expanduser()

    table_png = str(out_prefix) + "_table.png"
    frames_png = str(out_prefix) + "_frames.png"

    with h5py.File(ep_path, "r") as h5:
        datasets = collect_datasets(h5)

        image_key = find_image_dataset(datasets)
        qpos_key = find_dataset(datasets, keywords=["qpos", "joint", "state"], ndim=2)
        action_key = find_dataset(datasets, keywords=["action"], ndim=2)
        instruction_key = find_dataset(datasets, keywords=["instruction", "text", "language", "prompt"], ndim=None)

        images = np.array(h5[image_key]) if image_key else None
        qpos = np.array(h5[qpos_key]) if qpos_key else None
        action = np.array(h5[action_key]) if action_key else None
        instruction = np.array(h5[instruction_key]) if instruction_key else None

        T = None
        for arr in [images, qpos, action]:
            if arr is not None and arr.ndim >= 1:
                T = arr.shape[0]
                break
        if T is None:
            T = 0

        qpos_dim = qpos.shape[-1] if qpos is not None and qpos.ndim >= 2 else None
        action_dim = action.shape[-1] if action is not None and action.ndim >= 2 else None

        # ---------- TABLE DATA ----------
        rows = [
            ["file", str(ep_path)],
            ["num_datasets", len(datasets)],
            ["T (steps)", T],
            ["image_key", image_key],
            ["qpos_key", qpos_key],
            ["action_key", action_key],
            ["instruction_key", instruction_key],
            ["images", stat_line(images)],
            ["qpos", stat_line(qpos)],
            ["action", stat_line(action)],
            ["qpos_dim", qpos_dim],
            ["action_dim", action_dim],
            ["instruction", short_text(instruction)],
        ]

        # ---------- SAVE TABLE IMAGE ----------
        fig_h = max(6, 0.5 * len(rows))
        fig, ax = plt.subplots(figsize=(16, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=[[r[0], str(r[1])] for r in rows],
            colLabels=["Field", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_height(0.08)

        plt.title(f"HDF5 EPISODE TABLE\n{ep_path.name}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(table_png, dpi=180, bbox_inches="tight")
        plt.close(fig)

        # ---------- FRAME IDS ----------
        frame_ids = []
        if images is not None and images.ndim >= 4 and images.shape[0] > 0:
            n = images.shape[0]
            if args.num_frames == 3:
                frame_ids = [0, n // 2, n - 1]
            else:
                frame_ids = [0, n // 4, n // 2, (3 * n) // 4, n - 1]

        # ---------- SAVE FRAMES IMAGE ----------
        if frame_ids:
            num = len(frame_ids)
            fig, axes = plt.subplots(1, num, figsize=(4 * num, 4.5))
            if num == 1:
                axes = [axes]

            for ax, fid in zip(axes, frame_ids):
                img = normalize_image(images[fid])
                ax.imshow(img)
                ax.set_title(f"Frame {fid}", fontsize=12)
                ax.axis("off")

            plt.suptitle(f"HDF5 EPISODE FRAMES\n{ep_path.name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(frames_png, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "No image dataset found", ha="center", va="center", fontsize=16)
            plt.tight_layout()
            plt.savefig(frames_png, dpi=180, bbox_inches="tight")
            plt.close(fig)

    print("saved table :", table_png)
    print("saved frames:", frames_png)


if __name__ == "__main__":
    main()
