import os
import glob
import numpy as np

SRC = "dataset_push_real_v5_trim_manual"
OUT = "data/niryo_push_real_v5_trim_minivla.npz"

INSTRUCTION = "push the object"
TEXT_LEN = 16

def encode_text(text, vocab, text_len=16):
    words = text.lower().strip().split()
    ids = []
    for w in words:
        ids.append(vocab.get(w, vocab["<unk>"]))
    ids = ids[:text_len]
    ids += [vocab["<pad>"]] * (text_len - len(ids))
    return np.array(ids, dtype=np.int64)

def main():
    files = sorted(glob.glob(os.path.join(SRC, "ep_*.npz")))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz found in {SRC}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "push": 2,
        "the": 3,
        "object": 4,
    }

    all_images = []
    all_states = []
    all_actions = []
    all_text_ids = []
    all_episode_ids = []
    all_step_ids = []

    lens = []
    skipped = []

    for ep_idx, path in enumerate(files):
        name = os.path.basename(path)
        d = np.load(path, allow_pickle=True)

        required = ["images", "joints", "actions_delta_joints", "valid_len"]
        missing = [k for k in required if k not in d.files]
        if missing:
            skipped.append((name, f"missing keys {missing}"))
            continue

        n = int(d["valid_len"])
        images = d["images"][:n]
        joints = d["joints"][:n].astype(np.float32)
        actions = d["actions_delta_joints"][:n].astype(np.float32)

        if images.ndim != 4 or images.shape[-1] != 3:
            skipped.append((name, f"bad images shape {images.shape}"))
            continue

        if joints.shape != (n, 6):
            skipped.append((name, f"bad joints shape {joints.shape}, valid_len={n}"))
            continue

        if actions.shape != (n, 6):
            skipped.append((name, f"bad actions shape {actions.shape}, valid_len={n}"))
            continue

        # Bỏ frame cuối của mỗi episode vì action cuối là 0 do không còn next step thật.
        # Train bằng transition thật: image_t + joint_t -> delta_joint_t.
        if n < 2:
            skipped.append((name, f"valid_len too short {n}"))
            continue

        use_n = n - 1

        img_use = images[:use_n].astype(np.uint8)
        state_use = joints[:use_n].astype(np.float32)
        action_use = actions[:use_n].astype(np.float32)

        text_id = encode_text(INSTRUCTION, vocab, TEXT_LEN)
        text_ids_use = np.repeat(text_id[None, :], use_n, axis=0)

        ep_ids_use = np.full((use_n,), ep_idx, dtype=np.int32)
        step_ids_use = np.arange(use_n, dtype=np.int32)

        all_images.append(img_use)
        all_states.append(state_use)
        all_actions.append(action_use)
        all_text_ids.append(text_ids_use)
        all_episode_ids.append(ep_ids_use)
        all_step_ids.append(step_ids_use)

        lens.append(use_n)

    if not all_images:
        raise RuntimeError("No valid data converted.")

    images = np.concatenate(all_images, axis=0).astype(np.uint8)
    states = np.concatenate(all_states, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    text_ids = np.concatenate(all_text_ids, axis=0).astype(np.int64)
    episode_ids = np.concatenate(all_episode_ids, axis=0).astype(np.int32)
    step_ids = np.concatenate(all_step_ids, axis=0).astype(np.int32)

    print("===== CONVERT SUMMARY =====")
    print("source:", SRC)
    print("output:", OUT)
    print("episodes found:", len(files))
    print("episodes converted:", len(lens))
    print("samples:", len(images))
    print("episode transition len min/mean/max:", min(lens), sum(lens)/len(lens), max(lens))
    print("images:", images.shape, images.dtype)
    print("states:", states.shape, states.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("text_ids:", text_ids.shape, text_ids.dtype)
    print("vocab:", vocab)

    action_norm = np.linalg.norm(actions, axis=1)
    print("action norm min/mean/max:", float(action_norm.min()), float(action_norm.mean()), float(action_norm.max()))

    if skipped:
        print("\n===== SKIPPED =====")
        for item in skipped:
            print(item)
    else:
        print("\nNo skipped episodes.")

    np.savez_compressed(
        OUT,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=np.array(vocab, dtype=object),
        episode_ids=episode_ids,
        step_ids=step_ids,
        instruction=np.array(INSTRUCTION),
        source_dataset=np.array(SRC),
    )

    print("\nSaved:", OUT)

if __name__ == "__main__":
    main()
