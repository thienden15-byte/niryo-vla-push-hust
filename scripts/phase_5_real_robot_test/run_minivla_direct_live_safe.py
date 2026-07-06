import argparse
import time
import cv2
import numpy as np
import torch

from models.mini_vla.vla_direct_policy import VLADirectPolicy

try:
    from pyniryo import NiryoRobot
except Exception as e:
    raise RuntimeError("Không import được pyniryo. Kiểm tra môi trường Python/PyNiryo.") from e


def encode_text(text, vocab, text_len=16):
    words = text.lower().strip().split()
    ids = [vocab.get(w, vocab.get("<unk>", 1)) for w in words]
    ids = ids[:text_len]
    ids += [vocab.get("<pad>", 0)] * (text_len - len(ids))
    return np.array(ids, dtype=np.int64)


def load_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    vocab = ckpt["vocab"]
    vocab_size = max(vocab.values()) + 1

    model = VLADirectPolicy(
        vocab_size=vocab_size,
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        d_model=ckpt["d_model"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
    action_std = np.asarray(ckpt["action_std"], dtype=np.float32)

    return model, vocab, action_mean, action_std, ckpt


def open_camera(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{cam_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Camera /dev/video{cam_id} opened but cannot read frame")

    return cap


def predict_action(model, frame_bgr, joints, text_ids, action_mean, action_std, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)

    state = torch.tensor(joints, dtype=torch.float32).unsqueeze(0).to(device)
    text = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model.act(img, text, state)
        pred_norm = pred_norm.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Model direct học action normalized, phải đổi về delta joint thật.
    pred_real = pred_norm * action_std + action_mean
    pred_real = np.nan_to_num(pred_real, nan=0.0, posinf=0.0, neginf=0.0)

    return pred_real, pred_norm


def clamp_target_joints(q):
    # Giới hạn mềm để tránh vượt biên quá mạnh.
    # Nếu robot/PyNiryo còn giới hạn chặt hơn, PyNiryo sẽ tự báo lỗi.
    limits = np.array([
        [-2.90,  2.90],   # j1
        [-1.85,  0.65],   # j2
        [-1.35,  1.57],   # j3
        [-2.10,  2.10],   # j4
        [-1.90,  1.90],   # j5
        [-2.50,  2.50],   # j6
    ], dtype=np.float32)

    q = np.asarray(q, dtype=np.float32)
    q = np.clip(q, limits[:, 0], limits[:, 1])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="169.254.200.200")
    ap.add_argument("--cam", type=int, default=3)
    ap.add_argument("--ckpt", default="checkpoints/minivla_push_trim_v5_direct.pt")
    ap.add_argument("--instruction", default="push the object")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--velocity", type=int, default=4)
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--max-delta", type=float, default=0.008)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===== DIRECT miniVLA LIVE SAFE =====")
    print("device:", device)
    print("robot ip:", args.ip)
    print("camera:", args.cam)
    print("ckpt:", args.ckpt)
    print("instruction:", args.instruction)
    print("steps:", args.steps)
    print("velocity:", args.velocity)
    print("action_scale:", args.action_scale)
    print("max_delta per joint:", args.max_delta)
    print("execute:", args.execute)

    model, vocab, action_mean, action_std, ckpt = load_model(args.ckpt, device)
    text_ids = encode_text(args.instruction, vocab, text_len=16)

    print("Loaded model OK")
    print("model_type:", ckpt.get("model_type"))
    print("vocab:", vocab)
    print("action_mean:", np.round(action_mean, 6))
    print("action_std :", np.round(action_std, 6))
    print("text_ids:", text_ids)

    cap = open_camera(args.cam)

    robot = None
    try:
        robot = NiryoRobot(args.ip)
        robot.set_learning_mode(False)
        robot.set_arm_max_velocity(args.velocity)

        print("\nRobot connected OK")
        print("Current joints:", np.round(robot.get_joints(), 5))

        for step in range(args.steps):
            # Xả buffer camera một chút để lấy frame mới
            for _ in range(2):
                cap.read()

            ok, frame = cap.read()
            if not ok or frame is None:
                print("Cannot read camera frame. Stop.")
                break

            joints = np.asarray(robot.get_joints(), dtype=np.float32)

            pred_action, pred_norm = predict_action(
                model=model,
                frame_bgr=frame,
                joints=joints,
                text_ids=text_ids,
                action_mean=action_mean,
                action_std=action_std,
                device=device,
            )

            scaled = pred_action * args.action_scale
            clipped = np.clip(scaled, -args.max_delta, args.max_delta)
            target = clamp_target_joints(joints + clipped)

            print(f"\n--- step {step+1}/{args.steps} ---")
            print("joints      :", np.round(joints, 5))
            print("pred_action :", np.round(pred_action, 5), "norm=", round(float(np.linalg.norm(pred_action)), 5))
            print("scaled_clip :", np.round(clipped, 5), "norm=", round(float(np.linalg.norm(clipped)), 5))
            print("target      :", np.round(target, 5))

            if not args.no_window:
                vis = frame.copy()
                cv2.putText(vis, f"step {step+1}/{args.steps}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(vis, f"action_norm={np.linalg.norm(clipped):.4f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("miniVLA direct live", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord("q")]:
                    print("User stopped by key.")
                    break

            if args.execute:
                robot.move_joints(target.tolist())
            else:
                print("DRY RUN only. Add --execute to move robot.")

            time.sleep(args.sleep)

        print("\nDONE")

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if robot is not None:
            try:
                robot.close_connection()
            except Exception:
                pass


if __name__ == "__main__":
    main()
