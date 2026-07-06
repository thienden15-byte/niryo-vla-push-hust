import os, sys, time, pickle, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/TinyVLA"))
sys.path.insert(0, os.path.expanduser("~/TinyVLA/llava-pythia"))

from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.conversation import conv_templates
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_pythia.mm_utils import tokenizer_image_token

base_live_path = Path.home() / "tinyvla_niryo_runtime/scripts/run_author10d_fixed50_xyz_chunk_live.py"
spec = importlib.util.spec_from_file_location("base_live", base_live_path)
base_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_live)


def decode_minmax(raw, stats):
    action_min = stats["action_min"].reshape(1, 1, -1)
    action_max = stats["action_max"].reshape(1, 1, -1)
    return ((raw + 1.0) / 2.0) * (action_max - action_min) + action_min


def move_joints_compat(robot, joints):
    try:
        robot.move_joints(*joints.tolist())
    except TypeError:
        robot.move_joints(joints.tolist())


def clip_vec(v, max_norm):
    n = float(np.linalg.norm(v))
    if n <= max_norm:
        return v, n, False
    return v / n * max_norm, n, True


def temporal_aggregate(all_time_actions_raw, t, k=0.01):
    actions_for_t = all_time_actions_raw[:, t, :]
    valid = np.any(np.abs(actions_for_t) > 1e-9, axis=1)
    populated = actions_for_t[valid]
    if len(populated) == 0:
        return None, 0
    # ưu tiên prediction mới hơn: hàng cuối mới hơn
    weights = np.exp(-k * np.arange(len(populated))[::-1])
    weights = weights / weights.sum()
    return np.sum(populated * weights[:, None], axis=0), len(populated)


def make_prompt(tokenizer, model, instruction, device):
    conv = conv_templates["pythia"].copy()
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + instruction
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + instruction
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " <|endoftext|>"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return input_ids, attention_mask


def infer_raw_chunk(frame_bgr, joints6, tokenizer, model, image_processor, input_ids, attention_mask, stats):
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    qpos7 = np.concatenate([joints6.astype(np.float32), np.array([0.0], dtype=np.float32)], axis=0)
    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos7 - qpos_mean) / qpos_std_safe

    robot_state = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(model_device, dtype=model_dtype)

    curr_image = base_live.frame_to_model_tensor(frame_bgr)
    image_tensor, image_tensor_r = base_live.preprocess_image(
        curr_image, image_processor, model_device, model_dtype
    )

    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        images_r=image_tensor_r,
        states=robot_state,
    )

    with torch.no_grad():
        raw = model(**batch, eval=True)

    return raw.detach().float().cpu().numpy()[0]


def find_safe_ik_candidate(robot, current_joints, current_pose, target_xyz_raw, args):
    current_xyz = current_pose[:3]
    current_rpy = current_pose[3:6]

    base_dxyz = target_xyz_raw - current_xyz

    candidates = []
    for y_gain in args.y_gains:
        d = base_dxyz.copy()
        d[1] *= y_gain

        for max_step in args.step_candidates:
            d_clip, raw_norm, clipped = clip_vec(d, max_step)
            target_pose = current_pose.copy()
            target_pose[:3] = current_xyz + d_clip
            target_pose[3:6] = current_rpy

            x, y, z = target_pose[:3]
            if z < args.min_z or x < 0.05 or x > args.max_x or abs(y) > args.max_abs_y:
                continue

            try:
                ik = np.array(robot.inverse_kinematics(target_pose.tolist()), dtype=np.float64)
            except Exception as e:
                candidates.append((False, y_gain, max_step, d_clip, None, f"IK_FAIL {repr(e)}"))
                continue

            dj = ik - current_joints
            max_all = float(np.max(np.abs(dj)))
            max_wrist = float(np.max(np.abs(dj[3:6])))

            ok = (max_all <= args.max_joint_delta) and (max_wrist <= args.max_wrist_delta)
            reason = f"max_all={max_all:.4f} max_wrist={max_wrist:.4f} dj456={np.round(dj[3:6], 4)}"
            candidates.append((ok, y_gain, max_step, d_clip, ik, reason))

            if ok:
                return True, y_gain, max_step, d_clip, ik, reason, candidates

    return False, None, None, None, None, "NO_SAFE_CANDIDATE", candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")
    p.add_argument("--instruction", default="push the green object to the right")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--velocity", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.35)
    p.add_argument("--temporal-k", type=float, default=0.01)

    p.add_argument("--min-z", type=float, default=0.055)
    p.add_argument("--max-x", type=float, default=0.320)
    p.add_argument("--max-abs-y", type=float, default=0.180)
    p.add_argument("--max-joint-delta", type=float, default=0.30)
    p.add_argument("--max-wrist-delta", type=float, default=0.25)

    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if not args.execute or args.confirm != "YES_AUTHOR_STYLE_SAFE":
        raise RuntimeError("Need --execute --confirm YES_AUTHOR_STYLE_SAFE")

    # Ưu tiên giữ đúng hướng model trước. Nếu IK nguy hiểm, tự giảm Y / giảm step.
    args.y_gains = [1.0, 0.75, 0.5, 0.25, 0.0]
    args.step_candidates = [0.008, 0.006, 0.004]

    ckpt = Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
    model_base = Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== AUTHOR-STYLE NIRYO SAFE ROLLOUT =====")
    print("instruction:", args.instruction)
    print("steps:", args.steps)
    print("NOTE: minmax decode + temporal aggregation + IK guard + move_joints")
    print("NOTE: orientation-mode = keep_current, rot6d NOT used")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print("\n===== CAMERA OPEN =====")
    cam_idx, cam_name, candidates = base_live.find_usb_camera()
    print("camera candidates:", candidates)
    print("auto selected:", cam_idx, cam_name)

    live = base_live.LiveCamera(cam_idx)
    live.start()

    print("\n===== ROBOT CONNECT =====")
    from pyniryo import NiryoRobot
    robot = NiryoRobot(args.ip)
    robot.set_learning_mode(False)
    robot.set_arm_max_velocity(args.velocity)

    print("\n===== LOAD MODEL ONCE =====")
    torch.cuda.empty_cache()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=str(ckpt),
        model_base=str(model_base),
        model_name="llava-pythia-lora",
        device="cuda",
        device_map="cuda",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    input_ids, attention_mask = make_prompt(tokenizer, model, args.instruction, device)

    chunk_size = int(getattr(model.config, "chunk_size", 16))
    action_dim = int(getattr(model.config, "action_dim", 10))
    all_time_actions_raw = np.zeros((args.steps, args.steps + chunk_size, action_dim), dtype=np.float32)

    print("model loaded OK")
    print("action_head_type:", getattr(model.config, "action_head_type", None))
    print("chunk_size:", chunk_size)
    print("action_dim:", action_dim)

    try:
        for t in range(args.steps):
            if live.should_stop():
                print("STOP: user pressed q")
                break

            print(f"\n===== ROLLOUT STEP {t+1}/{args.steps} =====")

            frame_bgr = live.get_frame(wait=True)

            current_joints = np.array(robot.get_joints(), dtype=np.float64)
            current_pose = base_live.pose_to_list(robot.get_pose())
            current_pose = np.array(current_pose, dtype=np.float64)
            current_xyz = current_pose[:3]

            raw_chunk = infer_raw_chunk(
                frame_bgr,
                current_joints,
                tokenizer,
                model,
                image_processor,
                input_ids,
                attention_mask,
                stats,
            )

            all_time_actions_raw[t, t:t + chunk_size, :] = raw_chunk

            agg_raw, votes = temporal_aggregate(all_time_actions_raw, t, k=args.temporal_k)
            if agg_raw is None:
                print("No aggregated action")
                continue

            decoded = decode_minmax(agg_raw.reshape(1, 1, -1), stats)[0, 0]
            target_xyz_raw = decoded[:3].astype(np.float64)
            raw_dxyz = target_xyz_raw - current_xyz

            print("current xyz:", np.round(current_xyz, 6))
            print("agg votes:", votes)
            print("target xyz raw:", np.round(target_xyz_raw, 6))
            print("raw dxyz:", np.round(raw_dxyz, 6), "norm:", round(float(np.linalg.norm(raw_dxyz)), 6))

            ok, y_gain, max_step, d_clip, ik_target, reason, candidates = find_safe_ik_candidate(
                robot, current_joints, current_pose, target_xyz_raw, args
            )

            print("candidate summary:")
            for c in candidates[:6]:
                c_ok, c_y, c_step, c_d, c_ik, c_reason = c
                print(f"  {'OK ' if c_ok else 'BAD'} y_gain={c_y} step={c_step} dxyz={np.round(c_d, 5)} {c_reason}")

            if not ok:
                print("STOP: no safe IK candidate")
                break

            print("SELECTED:")
            print("y_gain:", y_gain, "max_step:", max_step)
            print("final dxyz:", np.round(d_clip, 6), "norm:", round(float(np.linalg.norm(d_clip)), 6))
            print("IK reason:", reason)
            print("moving by joints...")

            before_joints = current_joints.copy()
            before_pose = current_pose.copy()

            move_joints_compat(robot, ik_target)
            time.sleep(args.sleep)

            new_joints = np.array(robot.get_joints(), dtype=np.float64)
            new_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)

            print("actual delta joints:", np.round(new_joints - before_joints, 6))
            print("new xyz:", np.round(new_pose[:3], 6))
            print("actual dxyz:", np.round(new_pose[:3] - before_pose[:3], 6))

            live.update_status([
                f"AUTHOR SAFE {t+1}/{args.steps}",
                f"cur={np.round(current_xyz, 3)}",
                f"raw_target={np.round(target_xyz_raw, 3)}",
                f"move={np.round(d_clip, 3)} yg={y_gain}",
            ])

        print("\nAUTHOR-STYLE SAFE ROLLOUT DONE")

    finally:
        try:
            robot.close_connection()
        except Exception:
            pass
        try:
            live.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
