import os, sys, time, pickle, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch

TINYVLA_REPO = Path(
    os.environ.get("TINYVLA_REPO", str(Path.home() / "TinyVLA"))
).expanduser()

sys.path.insert(0, str(TINYVLA_REPO))
sys.path.insert(0, str(TINYVLA_REPO / "llava-pythia"))

# Reuse DiffIK runtime đã chạy được ở Step 30/31
diffik_path = (
    Path(__file__).resolve().parents[3]
    / "scripts/phase_5_real_robot_test/tiny_vla/"
      "run_author_style_niryo_diffik_rollout.py"
)
spec = importlib.util.spec_from_file_location("diffik_runtime", diffik_path)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

from llava_pythia.model.builder import load_pretrained_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="169.254.200.200")
    p.add_argument("--instruction", default="push the green object to the right")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--velocity", type=int, default=5)
    p.add_argument("--sleep", type=float, default=0.22)

    # đi vừa tới vật vừa hạ Z
    p.add_argument("--descend-start-x", type=float, default=0.155)
    p.add_argument("--contact-z", type=float, default=0.085)
    p.add_argument("--min-z", type=float, default=0.078)
    p.add_argument("--z-descend-step", type=float, default=0.003)

    # di chuyển theo model nhưng giới hạn an toàn
    p.add_argument("--max-cart-step", type=float, default=0.012)
    p.add_argument("--max-dq", type=float, default=0.075)
    p.add_argument("--temporal-k", type=float, default=0.01)
    p.add_argument("--y-gain", type=float, default=0.7)

    # sau khi xuống đủ thấp thì đẩy
    p.add_argument("--push-step", type=float, default=0.008)
    p.add_argument("--push-repeat", type=int, default=10)

    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if not args.execute or args.confirm != "YES_APPROACH_DESCEND_PUSH":
        raise RuntimeError("Need --execute --confirm YES_APPROACH_DESCEND_PUSH")

    ckpt = Path(
        os.environ.get(
            "TINYVLA_MODEL_PATH",
            str(
                Path.home()
                / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
            ),
        )
    ).expanduser()
    model_base = Path(
        os.environ.get(
            "TINYVLA_MODEL_BASE",
            str(
                Path.home()
                / "TinyVLA/pretrained/Llava-Pythia-400M"
            ),
        )
    ).expanduser()
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== HYBRID APPROACH + DESCEND + PUSH =====")
    print("Model guides X/Y. Runtime forces Z ramp while approaching.")
    print("contact_z:", args.contact_z)
    print("descend_start_x:", args.descend_start_x)
    print("z_descend_step:", args.z_descend_step)
    print("max_cart_step:", args.max_cart_step)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print("\n===== CAMERA OPEN =====")
    cam_idx, cam_name, candidates = R.base_live.find_usb_camera()
    print("camera candidates:", candidates)
    print("auto selected:", cam_idx, cam_name)

    live = R.base_live.LiveCamera(cam_idx)
    live.start()

    from pyniryo import NiryoRobot

    robot = NiryoRobot(args.ip)

    try:
        robot.set_learning_mode(False)
        robot.set_arm_max_velocity(args.velocity)

        print("\n===== MOVE HOME =====")
        robot.move_to_home_pose()
        time.sleep(0.8)

        home_pose = np.array(R.pose_to_list(robot.get_pose()), dtype=float)
        print("home pose:", np.round(home_pose, 6))

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
        input_ids, attention_mask = R.make_prompt(tokenizer, model, args.instruction, device)

        chunk_size = int(getattr(model.config, "chunk_size", 16))
        action_dim = int(getattr(model.config, "action_dim", 10))

        all_time_actions_raw = np.zeros(
            (args.steps, args.steps + chunk_size, action_dim),
            dtype=np.float32,
        )

        push_count = 0

        for t in range(args.steps):
            if live.should_stop():
                print("STOP: q pressed")
                break

            frame_bgr = live.get_frame(wait=True)

            current_joints = np.array(robot.get_joints(), dtype=np.float64)
            current_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=np.float64)
            current_xyz = current_pose[:3]

            # Nếu đã đủ thấp thì chuyển sang push +X
            if current_xyz[2] <= args.contact_z:
                mode = "push"
                raw_dxyz = np.array([args.push_step, 0.0, 0.0], dtype=float)
                push_count += 1

                if push_count > args.push_repeat:
                    print("PUSH DONE")
                    break

            else:
                mode = "approach_descend"

                raw_chunk = R.infer_raw_chunk(
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

                agg_raw, votes = R.temporal_aggregate(
                    all_time_actions_raw,
                    t,
                    k=args.temporal_k,
                )

                if agg_raw is None:
                    print("No aggregated action")
                    continue

                decoded = R.decode_minmax(agg_raw.reshape(1, 1, -1), stats)[0, 0]
                target_xyz_raw = decoded[:3].astype(np.float64)

                raw_dxyz = target_xyz_raw - current_xyz

                # giảm Y vì Niryo IK nhạy với Y
                raw_dxyz[1] *= args.y_gain

                # Quan trọng: vừa tới vật vừa hạ Z.
                # Không hạ thẳng tại chỗ nữa.
                if current_xyz[0] >= args.descend_start_x:
                    forced_dz = -min(args.z_descend_step, current_xyz[2] - args.contact_z)
                    raw_dxyz[2] = min(raw_dxyz[2], forced_dz)
                else:
                    # lúc mới rời home chỉ hạ nhẹ, tránh chọc xuống quá sớm
                    raw_dxyz[2] = min(raw_dxyz[2], -0.001)

            dxyz, raw_norm, cart_clipped = R.clip_vec(raw_dxyz, args.max_cart_step)

            next_xyz = current_xyz + dxyz

            if next_xyz[2] < args.min_z:
                print("STOP: next z below min_z")
                break

            if next_xyz[0] > 0.34:
                print("STOP: x too far")
                break

            J, fk_pose = R.numerical_xyz_jacobian(robot, current_joints)
            dq, pred_dxyz, dq_clipped = R.solve_diffik(
                J,
                dxyz,
                max_dq=args.max_dq,
            )

            target_joints = current_joints + dq

            print(f"\n===== STEP {t+1}/{args.steps} =====")
            print("mode:", mode)
            print("current xyz:", np.round(current_xyz, 6))
            print("raw dxyz:", np.round(raw_dxyz, 6))
            print("used dxyz:", np.round(dxyz, 6), "cart_clipped:", cart_clipped)
            print("dq:", np.round(dq, 6), "dq_clipped:", dq_clipped)
            print("wrist dq j4/j5/j6:", np.round(dq[3:6], 6))

            live.update_status([
                f"{mode.upper()} {t+1}/{args.steps}",
                f"cur={np.round(current_xyz, 3)}",
                f"move={np.round(dxyz, 3)}",
                f"contact_z={args.contact_z:.3f}",
                "q stop",
            ])

            before_pose = current_pose.copy()

            R.move_joints_compat(robot, target_joints)
            time.sleep(args.sleep)

            new_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=np.float64)

            print("new xyz:", np.round(new_pose[:3], 6))
            print("actual dxyz:", np.round(new_pose[:3] - before_pose[:3], 6))

        final_pose = np.array(R.base_live.pose_to_list(robot.get_pose()), dtype=float)
        print("\nfinal pose:", np.round(final_pose, 6))
        print("HYBRID APPROACH DESCEND PUSH DONE")

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
