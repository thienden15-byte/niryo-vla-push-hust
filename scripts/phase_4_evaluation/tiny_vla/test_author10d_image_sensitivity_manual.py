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



def summarize_actions(name, decoded_chunk, current_xyz):
    xyz = decoded_chunk[:, :3]
    dxyz = xyz - current_xyz.reshape(1, 3)

    print(f"\n===== RESULT: {name} =====")
    print("xyz first 5:")
    for i in range(min(5, len(xyz))):
        print(f"  {i+1:02d}: xyz={np.round(xyz[i], 6)} dxyz={np.round(dxyz[i], 6)}")

    print("xyz mean :", np.round(xyz.mean(axis=0), 6))
    print("xyz std  :", np.round(xyz.std(axis=0), 6))
    print("dxyz mean:", np.round(dxyz.mean(axis=0), 6))
    print("dxyz std :", np.round(dxyz.std(axis=0), 6))
    print("z min/mean/max:", round(float(xyz[:,2].min()), 6), round(float(xyz[:,2].mean()), 6), round(float(xyz[:,2].max()), 6))
    return xyz, dxyz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")
    p.add_argument("--instruction", default="push the green object to the right")
    p.add_argument("--num-scenes", type=int, default=4)
    args = p.parse_args()

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

    print("===== TINYVLA IMAGE SENSITIVITY TEST: NO ROBOT MOVEMENT =====")
    print("This script loads model once, then compares predictions under different camera scenes.")
    print("NO move_joints, NO move_pose.")
    print("instruction:", args.instruction)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print("\n===== CAMERA OPEN =====")
    cam_idx, cam_name, candidates = base_live.find_usb_camera()
    print("camera candidates:", candidates)
    print("auto selected:", cam_idx, cam_name)

    live = base_live.LiveCamera(cam_idx)
    live.start()

    print("\n===== ROBOT READ ONLY CONNECT =====")
    from pyniryo import NiryoRobot
    robot = NiryoRobot(args.ip)
    robot.set_learning_mode(False)

    current_joints = np.array(robot.get_joints(), dtype=np.float64)
    current_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
    current_xyz = current_pose[:3]

    print("current joints:", np.round(current_joints, 6))
    print("current xyz:", np.round(current_xyz, 6))

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

    scene_names = [
        "NORMAL_OBJECT_VISIBLE",
        "NO_OBJECT_REMOVE_GREEN_CUBE",
        "COVER_CAMERA_OR_BLANK_VIEW",
        "OBJECT_MOVED_TO_DIFFERENT_POSITION",
    ]

    results = {}

    try:
        for i in range(args.num_scenes):
            name = scene_names[i] if i < len(scene_names) else f"SCENE_{i+1}"

            print("\n" + "="*70)
            print(f"SCENE {i+1}/{args.num_scenes}: {name}")
            print("="*70)

            if name == "NORMAL_OBJECT_VISIBLE":
                print("Set scene: keep green object visible in camera.")
            elif name == "NO_OBJECT_REMOVE_GREEN_CUBE":
                print("Set scene: remove green object from the table/camera view.")
            elif name == "COVER_CAMERA_OR_BLANK_VIEW":
                print("Set scene: cover camera or point it to a blank area.")
            elif name == "OBJECT_MOVED_TO_DIFFERENT_POSITION":
                print("Set scene: move green object to another clearly different position.")
            else:
                print("Set your custom scene.")

            ans = input("Press ENTER after setting this scene, or type q then ENTER to stop: ").strip()
            if ans.lower() == "q":
                break

            frame_bgr = live.get_frame(wait=True)

            # Giữ qpos/current_xyz cố định theo robot hiện tại để test riêng ảnh.
            current_joints = np.array(robot.get_joints(), dtype=np.float64)
            current_pose = np.array(base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
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

            decoded_chunk = decode_minmax(raw_chunk.reshape(1, chunk_size, action_dim), stats)[0]
            xyz, dxyz = summarize_actions(name, decoded_chunk, current_xyz)

            results[name] = {
                "xyz_mean": xyz.mean(axis=0),
                "xyz_std": xyz.std(axis=0),
                "dxyz_mean": dxyz.mean(axis=0),
                "first_xyz": xyz[0],
            }

            live.update_status([
                f"SENS {i+1}/{args.num_scenes}",
                name,
                f"mean xyz={np.round(xyz.mean(axis=0), 3)}",
            ])

        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        names = list(results.keys())
        for name in names:
            r = results[name]
            print(f"{name}:")
            print("  first_xyz:", np.round(r["first_xyz"], 6))
            print("  xyz_mean :", np.round(r["xyz_mean"], 6))
            print("  dxyz_mean:", np.round(r["dxyz_mean"], 6))

        if len(names) >= 2:
            base = results[names[0]]["xyz_mean"]
            print("\nDIFFERENCE FROM FIRST SCENE:")
            for name in names[1:]:
                diff = results[name]["xyz_mean"] - base
                print(f"{name} - {names[0]}:", np.round(diff, 6), "norm=", round(float(np.linalg.norm(diff)), 6))

        print("\nINTERPRETATION:")
        print("- If xyz_mean changes clearly when object/camera changes, model is using visual input.")
        print("- If xyz_mean is almost the same across scenes, model may rely mostly on qpos/average trajectory.")
        print("- This script DID NOT move the robot.")

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
