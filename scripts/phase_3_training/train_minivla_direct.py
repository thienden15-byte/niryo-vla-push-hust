
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.mini_vla.vla_direct_policy import VLADirectPolicy


class TrainingDataset(Dataset):
    def __init__(self, path, resize_to=224, normalize_action=True):
        data = np.load(path, allow_pickle=True)

        self.images = data["images"]
        self.states = data["states"].astype(np.float32)
        self.actions_raw = data["actions"].astype(np.float32)
        self.text_ids = data["text_ids"].astype(np.int64)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]

        self.resize_to = resize_to
        self.normalize_action = normalize_action

        self.action_mean = self.actions_raw.mean(axis=0).astype(np.float32)
        self.action_std = self.actions_raw.std(axis=0).astype(np.float32)
        self.action_std = np.maximum(self.action_std, 1e-6).astype(np.float32)

        if normalize_action:
            self.actions = ((self.actions_raw - self.action_mean) / self.action_std).astype(np.float32)
        else:
            self.actions = self.actions_raw

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.cv2 is not None and (img.shape[0] != self.resize_to or img.shape[1] != self.resize_to):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        text_ids = torch.from_numpy(self.text_ids[idx]).long()

        return img, state, action, text_ids


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=str, required=True)
    ap.add_argument("--resize-to", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--save-path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = TrainingDataset(
        args.dataset_path,
        resize_to=args.resize_to,
        normalize_action=True,
    )

    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions_raw.shape[1]

    model = VLADirectPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
    ).to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0

        for img, state, action, text_ids in loader:
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            pred = model(img, text_ids, state)
            loss = torch.nn.functional.mse_loss(pred, action)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * img.size(0)

        avg = total / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}  norm_mse={avg:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": args.d_model,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
            "normalize_action": True,
            "model_type": "direct_mse",
        },
        args.save_path,
    )

    print("Saved checkpoint:", args.save_path)


if __name__ == "__main__":
    main()
