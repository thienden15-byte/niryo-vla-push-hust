import argparse
from pathlib import Path
import h5py
import numpy as np
import cv2


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
        score = 0

        if len(shape) == 4:
            if shape[-1] == 3:      # T,H,W,3
                score += 10
            elif shape[1] == 3:     # T,3,H,W
                score += 8

            if "image" in lname or "camera" in lname or "cam" in lname or "rgb" in lname:
                score += 10

        if score > 0:
            cands.append((score, name))

    if not cands:
        return None

    cands.sort(reverse=True)
    return cands[0][1]


def normalize_image(img):
    arr = np.array(img)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True, help="Path tới episode .hdf5/.h5")
    ap.add_argument("--out-dir", required=True, help="Thư mục xuất frame")
    ap.add_argument("--every", type=int, default=1, help="Lấy mỗi N frame. every=1 là lấy tất cả")
    ap.add_argument("--max-frames", type=int, default=9999, help="Giới hạn số frame xuất")
    args = ap.parse_args()

    ep_path = Path(args.ep).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(ep_path, "r") as h5:
        datasets = collect_datasets(h5)
        image_key = find_image_key(datasets)

        if image_key is None:
            print("Không tìm thấy image dataset.")
            print("Các dataset trong file:")
            for name, shape, dtype in datasets:
                print(name, shape, dtype)
            raise SystemExit(1)

        images = h5[image_key]
        T = images.shape[0]

        print("episode:", ep_path)
        print("image_key:", image_key)
        print("image_shape:", images.shape)
        print("T:", T)
        print("out_dir:", out_dir)

        count = 0
        for i in range(0, T, args.every):
            if count >= args.max_frames:
                break

            img = normalize_image(images[i])

            # matplotlib/PIL dùng RGB, OpenCV ghi file cần BGR
            if img.ndim == 3 and img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img

            out_path = out_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(out_path), img_bgr)

            count += 1

        print(f"saved {count} frames to:", out_dir)


if __name__ == "__main__":
    main()
