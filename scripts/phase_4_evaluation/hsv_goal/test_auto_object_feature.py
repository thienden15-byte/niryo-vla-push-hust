#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_auto(img224, args):
    h_img, w_img = img224.shape[:2]

    x1 = int(args.roi_x1 * w_img)
    x2 = int(args.roi_x2 * w_img)
    y1 = int(args.roi_y1 * h_img)
    y2 = int(args.roi_y2 * h_img)

    hsv = cv2.cvtColor(img224, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Vật thường có màu nổi hơn nền bàn: saturation cao hơn.
    mask = (
        (h >= args.h_min) &
        (h <= args.h_max) &
        (s >= args.s_min) &
        (v >= args.v_min) &
        (v <= args.v_max)
    ).astype(np.uint8) * 255

    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = float(cv2.contourArea(c))

        if area < args.min_area_px:
            continue

        if area > args.max_area_px:
            continue

        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        x, y, ww, hh = cv2.boundingRect(c)

        if ww <= 1 or hh <= 1:
            continue

        ratio = max(ww / hh, hh / ww)
        if ratio > args.max_ratio:
            continue

        # Ưu tiên vật có diện tích lớn vừa phải, nằm trong ROI.
        score = area

        candidates.append((score, area, c, cx, cy, x, y, ww, hh))

    if not candidates:
        return None, mask

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, area, c, cx, cy, x, y, ww, hh = candidates[0]

    cx_norm = cx / float(w_img)
    cy_norm = cy / float(h_img)
    area_norm = area / float(w_img * h_img)

    result = {
        "feature": np.array([cx_norm, cy_norm, area_norm], dtype=np.float32),
        "contour": c,
        "cx": cx,
        "cy": cy,
        "bbox": (x, y, ww, hh),
        "area": area,
    }

    return result, mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera-index", type=int, default=3)

    # HSV trong OpenCV: H từ 0 đến 179.
    # Xanh lá thường nằm khoảng H=35..90.
    parser.add_argument("--h-min", type=int, default=35)
    parser.add_argument("--h-max", type=int, default=90)

    parser.add_argument("--s-min", type=int, default=25)
    parser.add_argument("--v-min", type=int, default=40)
    parser.add_argument("--v-max", type=int, default=255)

    parser.add_argument("--min-area-px", type=float, default=25.0)
    parser.add_argument("--max-area-px", type=float, default=2500.0)
    parser.add_argument("--max-ratio", type=float, default=4.0)

    # ROI: vùng làm việc trên ảnh 224x224.
    # Nếu detect nhầm robot/bàn, ta chỉnh 4 tham số này sau.
    parser.add_argument("--roi-x1", type=float, default=0.05)
    parser.add_argument("--roi-x2", type=float, default=0.95)
    parser.add_argument("--roi-y1", type=float, default=0.05)
    parser.add_argument("--roi-y2", type=float, default=0.95)

    args = parser.parse_args()

    print("=" * 100)
    print("TEST AUTO OBJECT FEATURE")
    print("=" * 100)
    print("Không cần click.")
    print("Camera tự tìm vật theo màu nổi bật / saturation.")
    print("SPACE: lưu feature hiện tại")
    print("q    : thoát")
    print("=" * 100)
    print("Args:", args)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Không mở được camera index:", args.camera_index)
        return

    Path("goal_v1/logs").mkdir(parents=True, exist_ok=True)

    last_feature = None
    last_vis = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Không đọc được frame.")
            continue

        img224 = cv2.resize(frame, (224, 224))
        vis = img224.copy()

        result, mask = detect_auto(img224, args)

        h_img, w_img = img224.shape[:2]
        rx1 = int(args.roi_x1 * w_img)
        rx2 = int(args.roi_x2 * w_img)
        ry1 = int(args.roi_y1 * h_img)
        ry2 = int(args.roi_y2 * h_img)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        if result is not None:
            feature = result["feature"]
            x, y, ww, hh = result["bbox"]
            cx = result["cx"]
            cy = result["cy"]

            last_feature = feature
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            txt1 = f"cx={feature[0]:.3f} cy={feature[1]:.3f} area={feature[2]:.5f}"
            txt2 = f"area_px={result['area']:.1f}"
            cv2.putText(vis, txt1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(vis, txt2, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "NO OBJECT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        show = np.hstack([vis, mask_bgr])

        cv2.imshow("AUTO DETECT | left=image right=mask", show)
        last_vis = show.copy()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            if last_feature is None:
                print("⚠️ Chưa detect được vật.")
            else:
                print("\nDetected object_feature [cx, cy, area]:", last_feature)
                cv2.imwrite("goal_v1/logs/auto_detect_last.png", last_vis)
                print("Saved image: goal_v1/logs/auto_detect_last.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
