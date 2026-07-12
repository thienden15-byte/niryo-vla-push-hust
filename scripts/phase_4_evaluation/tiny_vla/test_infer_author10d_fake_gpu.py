import os
import sys
import pickle
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
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava_pythia.mm_utils import tokenizer_image_token


def expand2square(imgs_chw, background_color):
    # imgs_chw: [B, C, H, W], value 0..1
    b, c, h, w = imgs_chw.shape
    max_dim = max(h, w)
    imgs_np = imgs_chw.permute(0, 2, 3, 1).detach().cpu().numpy()

    expanded = np.full(
        (b, max_dim, max_dim, c),
        background_color,
        dtype=np.float32,
    )

    if h == w:
        expanded = imgs_np
    elif h > w:
        offset = (max_dim - w) // 2
        expanded[:, :h, offset:offset + w, :] = imgs_np
    else:
        offset = (max_dim - h) // 2
        expanded[:, offset:offset + h, :w, :] = imgs_np

    return torch.tensor(expanded, dtype=imgs_chw.dtype, device=imgs_chw.device)


def main():
    ckpt = Path(
        os.environ.get(
            "TINYVLA_MODEL_PATH",
            str(
                Path.home()
                / "tinyvla_niryo_ckpt/author_10d_full_5000steps"
            ),
        )
    ).expanduser()
    base = Path(
        os.environ.get(
            "TINYVLA_MODEL_BASE",
            str(
                Path.home()
                / "TinyVLA/pretrained/Llava-Pythia-400M"
            ),
        )
    ).expanduser()
    stats_path = ckpt / "dataset_stats.pkl"

    print("===== AUTHOR 10D FAKE INFERENCE: NO ROBOT, NO CAMERA =====")
    print("ckpt:", ckpt)
    print("base:", base)
    print("stats:", stats_path)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print("\n===== STATS SHAPES =====")
    for k in ["qpos_mean", "qpos_std", "action_mean", "action_std", "action_min", "action_max", "example_qpos"]:
        v = stats[k]
        print(k, getattr(v, "shape", None), type(v))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Fake GPU test needs CUDA first.")

    torch.cuda.empty_cache()

    print("\n===== LOAD MODEL =====")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=str(ckpt),
        model_base=str(base),
        model_name="llava-pythia-lora",
        device="cuda",
        device_map="cuda",
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    print("model loaded OK")
    print("device:", model_device)
    print("dtype:", model_dtype)
    print("context_len:", context_len)
    print("config state_dim:", getattr(model.config, "state_dim", None))
    print("config action_dim:", getattr(model.config, "action_dim", None))
    print("config chunk_size:", getattr(model.config, "chunk_size", None))

    print("\n===== FAKE IMAGE =====")
    # Dùng 2 ảnh giả để khớp input image + image_r.
    # Kích thước 180x320 gần thông báo train: Current Image Size [180,320]
    curr_image = torch.rand(2, 3, 180, 320, device=model_device, dtype=torch.float32)
    image, image_r = torch.chunk(curr_image, 2, dim=0)

    image = expand2square(image, tuple(x for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(
        image,
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"].to(model_device, dtype=model_dtype)

    image_r = expand2square(image_r, tuple(x for x in image_processor.image_mean))
    image_tensor_r = image_processor.preprocess(
        image_r,
        return_tensors="pt",
        do_normalize=True,
        do_rescale=False,
        do_center_crop=False,
    )["pixel_values"].to(model_device, dtype=model_dtype)

    print("image_tensor:", tuple(image_tensor.shape), image_tensor.dtype, image_tensor.device)
    print("image_tensor_r:", tuple(image_tensor_r.shape), image_tensor_r.dtype, image_tensor_r.device)

    print("\n===== FAKE 7D ROBOT STATE FROM DATASET STATS =====")
    qpos_raw = stats["example_qpos"][0].astype(np.float32)  # shape (7,)
    qpos_mean = stats["qpos_mean"].astype(np.float32)
    qpos_std = stats["qpos_std"].astype(np.float32)
    qpos_std_safe = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
    qpos_norm = (qpos_raw - qpos_mean) / qpos_std_safe

    print("qpos_raw:", np.round(qpos_raw, 6))
    print("qpos_norm:", np.round(qpos_norm, 6))
    print("qpos_norm shape:", qpos_norm.shape)

    robot_state = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(model_device, dtype=model_dtype)

    print("\n===== PROMPT =====")
    raw_lang = "push the green object to the right"
    conv = conv_templates["pythia"].copy()

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + raw_lang
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + raw_lang

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    prompt += " <|endoftext|>"

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model_device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    print("instruction:", raw_lang)
    print("input_ids shape:", tuple(input_ids.shape))

    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        images_r=image_tensor_r,
        states=robot_state,
    )

    print("\n===== RUN MODEL INFERENCE =====")
    with torch.no_grad():
        actions = model(**batch, eval=True)

    print("\n===== INFERENCE OK =====")
    print("actions type:", type(actions))
    print("actions shape:", tuple(actions.shape))
    print("actions dtype:", actions.dtype)
    print("actions device:", actions.device)

    actions_norm = actions.detach().float().cpu().numpy()

    action_mean = stats["action_mean"].reshape(1, 1, -1)
    action_std = stats["action_std"].reshape(1, 1, -1)
    action_min = stats["action_min"].reshape(1, 1, -1)
    action_max = stats["action_max"].reshape(1, 1, -1)

    actions_real = actions_norm * action_std + action_mean
    within = np.logical_and(actions_real >= action_min, actions_real <= action_max)

    print("\n===== ACTION 10D RESULT =====")
    print("normalized first action:")
    print(np.round(actions_norm[0, 0], 6))

    print("denormalized first action 10D:")
    print(np.round(actions_real[0, 0], 6))

    print("first 3 denormalized actions:")
    print(np.round(actions_real[0, :3], 6))

    print("within dataset range ratio:", float(within.mean()))
    print("real action min:", np.round(actions_real.min(axis=(0, 1)), 6))
    print("real action max:", np.round(actions_real.max(axis=(0, 1)), 6))

    print("\n10D meaning:")
    print("[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]")

    print("\nVRAM allocated MB:", round(torch.cuda.memory_allocated() / 1024 / 1024, 2))
    print("FAKE AUTHOR 10D INFERENCE TEST DONE")


if __name__ == "__main__":
    main()
