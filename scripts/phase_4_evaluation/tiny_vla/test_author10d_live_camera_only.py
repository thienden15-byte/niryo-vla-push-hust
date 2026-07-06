import cv2
import time
from pathlib import Path


def find_usb_camera():
    candidates = []
    for i in range(20):
        name_path = Path(f"/sys/class/video4linux/video{i}/name")
        if name_path.exists():
            name = name_path.read_text(errors="ignore").strip()
            candidates.append((i, name))
            if "USB Camera2" in name or "USB" in name:
                return i, name, candidates
    if candidates:
        return candidates[0][0], candidates[0][1], candidates
    return None, None, candidates


cam_idx, cam_name, candidates = find_usb_camera()

print("camera candidates:", candidates)
print("auto selected:", cam_idx, cam_name)

if cam_idx is None:
    raise RuntimeError("No camera found")

cap = cv2.VideoCapture(cam_idx)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {cam_idx}")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

window_name = "AUTHOR10D LIVE CAMERA - press q to quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

print()
print("Live camera window opened.")
print("Press q in the camera window to quit.")

frame_id = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARNING: cannot read frame")
            time.sleep(0.05)
            continue

        frame_id += 1

        show = frame.copy()
        cv2.putText(
            show,
            f"AUTHOR10D LIVE CAMERA | frame={frame_id} | press q to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("User quit live camera.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")
