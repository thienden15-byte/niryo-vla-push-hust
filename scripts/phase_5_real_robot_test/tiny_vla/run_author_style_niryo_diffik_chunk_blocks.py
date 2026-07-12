import os, sys, time, pickle, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch

TINYVLA_REPO = Path(
    os.environ.get("TINYVLA_REPO", str(Path.home() / "TinyVLA"))
).expanduser()

sys.path.insert(0, str(TINYVLA_REPO))
sys.path.insert(0, str(TINYVLA_REPO / "llava-pythia"))

from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.conversation import conv_templates
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_pythia.mm_utils import tokenizer_image_token

base_live_path = Path(__file__).resolve().parents[3] / "scripts/common/tiny_vla/run_author10d_fixed50_xyz_chunk_live.py"
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


def pose_to_list(pose):
    if hasattr(pose, "to_list"):
        return list(pose.to_list()[:6])
    if isinstance(pose, (list, tuple, np.ndarray)):
        return list(pose[:6])
    return [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]


def call_fk(robot, joints):
    joints = np.array(joints, dtype=float)
    errors = []
    try:
        return np.array(pose_to_list(robot.forward_kinematics(joints.tolist())), dtype=float)
    except Exception as e:
        errors.append(("list", repr(e)))
    try:
        return np.array(pose_to_list(robot.forward_kinematics(*joints.tolist())), dtype=float)
    except Exception as e:
        errors.append(("*args", repr(e)))
    raise RuntimeError("forward_kinematics failed: " + str(errors))


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
    image_tensor, image_tensor_r = base_live.preprocess_image(curr_image, image_processor, model_device, model_dtype)

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


def numerical_xyz_jacobian(robot, joints, eps=1e-2):
    joints = np.array(joints, dtype=float)
    base_pose = call_fk(robot, joints)
    base_xyz = base_pose[:3]

    J = np.zeros((3, 6), dtype=float)

    for i in range(6):
        jp = joints.copy()
        jp[i] += eps
        pp = call_fk(robot, jp)
        J[:, i] = (pp[:3] - base_xyz) / eps

    return J, base_pose


def solve_diffik(J, dxyz, max_dq=0.035, damping=0.03):
    """
    Solve: J dq ~= dxyz, nhưng phạt cổ tay để tránh joint4/5/6 xoay mạnh.
    """
    # Penalize wrist joints more.
    W = np.diag([1.0, 1.0, 1.2, 5.0, 5.0, 5.0])

    A = np.vstack([J, damping * W])
    b = np.concatenate([dxyz, np.zeros(6)])

    dq, *_ = np.linalg.lstsq(A, b, rcond=None)

    max_abs = float(np.max(np.abs(dq)))
    clipped = False
    if max_abs > max_dq:
        dq = dq / max_abs * max_dq
        clipped = True

    pred = J @ dq
    return dq, pred, clipped



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")
    p.add_argument("--instruction", default="push the green object to the right")

    # số block, mỗi block = 1 lần model dự đoán chunk 16 action
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--chunk-steps", type=int, default=16)

    p.add_argument("--velocity", type=int, default=6)
    p.add_argument("--sleep", type=float, default=0.18)

    # nới rộng hơn bản trước, nhưng vẫn còn guard
    p.add_argument("--max-cart-step", type=float, default=0.020)
    p.add_argument("--max-dq", type=float, default=0.055)
    p.add_argument("--min-z", type=float, default=0.080)
    p.add_argument("--max-x", type=float, default=0.360)
    p.add_argument("--max-abs-y", type=float, default=0.220)

    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if not args.execute or args.confirm != "YES_CHUNK_BLOCKS":
        raise RuntimeError("Need --execute --confirm YES_CHUNK_BLOCKS")

    ckpt = Path(
        os.environ.get(
            "TINYVLA_MODEL_PATH",
            str(Path.home() / "tinyvla_niryo_ckpt/author_10d_full_5000steps"),
        )
    ).expanduser()
    model_base = Path(
        os.environ.get(
            "TINYVLA_MODEL_BASE",
            str(Path.home() / "TinyVLA/pretrained/Llava-Pythia-400M"),
        )
    ).expanduser()
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== AUTHOR-STYLE NIRYO CHUNK BLOCKS ROLLOUT =====")
    print("Load model once.")
    print("Each block: observe once -> TinyVLA predicts 16-action diffusion chunk -> execute 16 actions.")
    print("After each block, wait for ENTER before next block.")
    print("rot6d not used; XYZ only; DiffIK + move_joints.")
    print("blocks:", args.blocks)
    print("chunk_steps:", args.chunk_steps)
    print("velocity:", args.velocity)
    print("max_cart_step:", args.max_cart_step)
    print("max_dq:", args.max_dq)
    print("min_z:", args.min_z)
    print("max_x:", args.max_x)
    print("max_abs_y:", args.max_abs_y)

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

    print("\n===== FK QUICK CHECK =====")
    cur_j = np.array(robot.get_joints(), dtype=np.float64)
    fk_pose = call_fk(robot, cur_j)
    real_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
    print("current joints:", np.round(cur_j, 6))
    print("FK pose:", np.round(fk_pose, 6))
    print("real pose:", np.round(real_pose, 6))
    print("FK-real xyz diff:", np.round(fk_pose[:3] - real_pose[:3], 6))

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

    print("model loaded OK")
    print("action_head_type:", getattr(model.config, "action_head_type", None))
    print("chunk_size:", chunk_size)
    print("action_dim:", action_dim)

    try:
        for b in range(args.blocks):
            print("\n" + "="*80)
            print(f"READY FOR BLOCK {b+1}/{args.blocks}")
            print("This will infer ONE 16-action chunk and execute it continuously.")
            ans = input("Press ENTER to run this 16-step block, or type q then ENTER to stop: ").strip()
            if ans.lower() == "q":
                print("Stopped by user.")
                break

            if live.should_stop():
                print("STOP: user pressed q in camera window")
                break

            print(f"\n===== BLOCK {b+1}/{args.blocks}: READ OBSERVATION =====")
            frame_bgr = live.get_frame(wait=True)
            obs_joints = np.array(robot.get_joints(), dtype=np.float64)
            obs_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
            obs_xyz = obs_pose[:3]

            print("obs joints:", np.round(obs_joints, 6))
            print("obs xyz:", np.round(obs_xyz, 6))

            print("\n===== RUN TINYVLA DIFFUSION ONCE =====")
            raw_chunk = infer_raw_chunk(
                frame_bgr,
                obs_joints,
                tokenizer,
                model,
                image_processor,
                input_ids,
                attention_mask,
                stats,
            )

            decoded_chunk = decode_minmax(raw_chunk.reshape(1, chunk_size, action_dim), stats)[0]

            n = min(args.chunk_steps, chunk_size)
            print("decoded_chunk shape:", decoded_chunk.shape)
            print("\nPREDICTED ACTION XYZ:")
            for i in range(n):
                print(f"  {i+1:02d}: xyz={np.round(decoded_chunk[i,:3], 6)}")

            print("\n===== EXECUTE CHUNK =====")

            for i in range(n):
                if live.should_stop():
                    print("STOP: user pressed q in camera window")
                    break

                print(f"\n--- BLOCK {b+1}, ACTION {i+1}/{n} ---")

                current_joints = np.array(robot.get_joints(), dtype=np.float64)
                current_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
                current_xyz = current_pose[:3]

                target_xyz_raw = decoded_chunk[i, :3].astype(np.float64)
                raw_dxyz = target_xyz_raw - current_xyz
                dxyz, raw_norm, cart_clipped = clip_vec(raw_dxyz, args.max_cart_step)

                target_xyz_safe = current_xyz + dxyz

                print("current xyz:", np.round(current_xyz, 6))
                print("target xyz raw:", np.round(target_xyz_raw, 6))
                print("raw dxyz:", np.round(raw_dxyz, 6), "norm:", round(float(np.linalg.norm(raw_dxyz)), 6))
                print("used dxyz:", np.round(dxyz, 6), "cart_clipped:", cart_clipped)
                print("target xyz safe:", np.round(target_xyz_safe, 6))

                if target_xyz_safe[2] < args.min_z:
                    print("STOP BLOCK: predicted z too low")
                    break
                if target_xyz_safe[0] < 0.05 or target_xyz_safe[0] > args.max_x:
                    print("STOP BLOCK: predicted x outside range")
                    break
                if abs(target_xyz_safe[1]) > args.max_abs_y:
                    print("STOP BLOCK: predicted y outside range")
                    break

                J, fk_pose = numerical_xyz_jacobian(robot, current_joints)
                dq, pred_dxyz, dq_clipped = solve_diffik(J, dxyz, max_dq=args.max_dq)

                target_joints = current_joints + dq
                pred_pose = call_fk(robot, target_joints)
                pred_xyz = pred_pose[:3]

                print("dq:", np.round(dq, 6), "dq_clipped:", dq_clipped)
                print("pred actual dxyz FK:", np.round(pred_xyz - current_xyz, 6))
                print("wrist dq j4/j5/j6:", np.round(dq[3:6], 6))

                if np.max(np.abs(dq)) > args.max_dq + 1e-6:
                    print("STOP BLOCK: dq too large")
                    break

                before_pose = current_pose.copy()
                before_joints = current_joints.copy()

                move_joints_compat(robot, target_joints)
                time.sleep(args.sleep)

                new_joints = np.array(robot.get_joints(), dtype=np.float64)
                new_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)

                print("actual delta joints:", np.round(new_joints - before_joints, 6))
                print("new xyz:", np.round(new_pose[:3], 6))
                print("actual dxyz:", np.round(new_pose[:3] - before_pose[:3], 6))

                live.update_status([
                    f"BLOCK {b+1}/{args.blocks} ACT {i+1}/{n}",
                    f"cur={np.round(current_xyz, 3)}",
                    f"raw={np.round(target_xyz_raw, 3)}",
                    f"move={np.round(dxyz, 3)}",
                ])

            print(f"\nBLOCK {b+1} DONE")

        print("\nCHUNK BLOCKS ROLLOUT DONE")

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
