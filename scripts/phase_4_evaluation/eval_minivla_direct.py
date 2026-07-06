import torch
import numpy as np
from models.mini_vla.vla_direct_policy import VLADirectPolicy

CKPT_PATH = "checkpoints/minivla_push_trim_v5_direct.pt"
DATA_PATH = "data/niryo_push_real_v5_trim_minivla.npz"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

try:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(CKPT_PATH, map_location=device)

data = np.load(DATA_PATH, allow_pickle=True)

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

print("Loaded Direct model OK")
print("model_type:", ckpt.get("model_type"))
print("state_dim:", ckpt["state_dim"])
print("action_dim:", ckpt["action_dim"])
print("d_model:", ckpt["d_model"])
print("action_mean:", np.round(action_mean, 6))
print("action_std :", np.round(action_std, 6))

images = data["images"]
states = data["states"].astype(np.float32)
actions = data["actions"].astype(np.float32)
text_ids = data["text_ids"]

N = len(images)
rng = np.random.default_rng(0)
idxs = rng.choice(N, size=min(300, N), replace=False)

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

        pred_norm = model.act(img, text, state)
        pred_norm_np = pred_norm.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Quan trọng: đổi action normalized về action thật
        pred_real = pred_norm_np * action_std + action_mean

        preds.append(pred_real)
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

dot = np.sum(preds * trues, axis=1)
cos = dot / ((np.linalg.norm(preds, axis=1) * np.linalg.norm(trues, axis=1)) + 1e-8)

print("\n===== DIRECT POLICY LOCAL OFFLINE EVAL 300 SAMPLES =====")
print("samples:", len(idxs))

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
