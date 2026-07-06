import numpy as np
import torch

from models.mini_vla.vla_diffusion_policy import VLADiffusionPolicy


CKPT_PATH = "checkpoints/minivla_push_trim_v5.pt"
DATA_PATH = "data/niryo_push_real_v5_trim_minivla.npz"
NUM_SAMPLES = 300
SEED = 0


def cosine_similarity(a, b, eps=1e-8):
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / den


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    data = np.load(DATA_PATH, allow_pickle=True)
    images = data["images"]
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    text_ids = data["text_ids"].astype(np.int64)

    print("dataset:", DATA_PATH)
    print("images:", images.shape, images.dtype)
    print("states:", states.shape, states.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("text_ids:", text_ids.shape, text_ids.dtype)

    try:
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(CKPT_PATH, map_location=device)

    vocab = ckpt["vocab"]
    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        d_model=ckpt["d_model"],
        diffusion_T=ckpt["diffusion_T"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\nLoaded Diffusion model OK")
    print("checkpoint:", CKPT_PATH)
    print("state_dim:", ckpt["state_dim"])
    print("action_dim:", ckpt["action_dim"])
    print("d_model:", ckpt["d_model"])
    print("diffusion_T:", ckpt["diffusion_T"])
    print("vocab:", vocab)

    N = len(images)
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(N, size=min(NUM_SAMPLES, N), replace=False)

    preds = []
    trues = []

    with torch.no_grad():
        for idx in idxs:
            img_np = images[idx]
            state_np = states[idx]
            true_action_np = actions[idx]
            text_np = text_ids[idx]

            img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(device)

            state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            text = torch.from_numpy(text_np).long().unsqueeze(0).to(device)

            pred = model.act(img, text, state)
            pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

            preds.append(pred_np)
            trues.append(true_action_np)

    preds = np.stack(preds)
    trues = np.stack(trues)

    err = np.linalg.norm(preds - trues, axis=1)
    true_norm = np.linalg.norm(trues, axis=1)
    pred_norm = np.linalg.norm(preds, axis=1)

    zero_pred = np.zeros_like(trues)
    zero_err = np.linalg.norm(zero_pred - trues, axis=1)

    mean_action = actions.mean(axis=0, keepdims=True)
    mean_pred = np.repeat(mean_action, len(trues), axis=0)
    mean_err = np.linalg.norm(mean_pred - trues, axis=1)

    cos = cosine_similarity(preds, trues)

    print("\n===== DIFFUSION POLICY LOCAL OFFLINE EVAL 300 SAMPLES SAME-IDX =====")
    print("samples:", len(idxs))
    print("sample selection: np.random.default_rng(0).choice(N, 300, replace=False)")

    print("\n--- norm ---")
    print("true norm mean:", float(true_norm.mean()))
    print("pred norm mean:", float(pred_norm.mean()))
    print("true norm min/max:", float(true_norm.min()), float(true_norm.max()))
    print("pred norm min/max:", float(pred_norm.min()), float(pred_norm.max()))

    print("\n--- error ---")
    print("model L2 error mean:", float(err.mean()))
    print("model L2 error median:", float(np.median(err)))
    print("zero baseline error mean:", float(zero_err.mean()))
    print("mean-action baseline error mean:", float(mean_err.mean()))

    print("\n--- cosine ---")
    print("cosine mean:", float(cos.mean()))
    print("cosine median:", float(np.median(cos)))
    print("cosine > 0 count:", int((cos > 0).sum()), "/", len(cos))

    print("\n--- per joint pred mean/std ---")
    print("pred mean:", np.round(preds.mean(axis=0), 5))
    print("pred std :", np.round(preds.std(axis=0), 5))

    print("\n--- per joint true mean/std ---")
    print("true mean:", np.round(trues.mean(axis=0), 5))
    print("true std :", np.round(trues.std(axis=0), 5))

    print("\nDONE")


if __name__ == "__main__":
    main()
