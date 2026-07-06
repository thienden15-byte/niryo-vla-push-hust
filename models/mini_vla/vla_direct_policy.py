
import torch
import torch.nn as nn

from .encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP


class VLADirectPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim, d_model=128):
        super().__init__()

        self.img_encoder = ImageEncoderTinyCNN(d_model=d_model)
        self.txt_encoder = TextEncoderTinyGRU(
            vocab_size=vocab_size,
            d_word=64,
            d_model=d_model,
        )
        self.state_encoder = StateEncoderMLP(
            state_dim=state_dim,
            d_model=d_model,
        )
        self.fusion = FusionMLP(d_model=d_model)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def encode_obs(self, image, text_tokens, state):
        img_token = self.img_encoder(image)
        txt_token = self.txt_encoder(text_tokens)
        state_token = self.state_encoder(state)
        fused = self.fusion(img_token, txt_token, state_token)
        return fused

    def forward(self, image, text_tokens, state):
        fused = self.encode_obs(image, text_tokens, state)
        return self.action_head(fused)

    def act(self, image, text_tokens, state):
        return self.forward(image, text_tokens, state)

    def loss(self, image, text_tokens, state, actions):
        pred = self.forward(image, text_tokens, state)
        return nn.functional.mse_loss(pred, actions)
