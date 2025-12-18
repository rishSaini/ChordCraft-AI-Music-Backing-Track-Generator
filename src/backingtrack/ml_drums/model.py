from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DrumModelConfig:
    n_voices: int = 9
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    steps: int = 16


class DrumTransformer(nn.Module):
    """
    Causal Transformer that predicts next-step drum hits (multi-label) + velocities.
    Input at each step = concatenated [hits, vels] from previous step.
    """
    def __init__(self, cfg: DrumModelConfig):
        super().__init__()
        self.cfg = cfg
        feat = cfg.n_voices * 2

        self.in_proj = nn.Linear(feat, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.steps, cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.hit_head = nn.Linear(cfg.d_model, cfg.n_voices)   # logits
        self.vel_head = nn.Linear(cfg.d_model, cfg.n_voices)   # sigmoid => 0..1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 16, 2V)
        returns:
          hit_logits: (B, 16, V)
          vel_pred:   (B, 16, V) in [0,1]
        """
        B, T, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :T, :]

        # causal mask: prevent attending to future
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.tr(h, mask=mask)

        hit_logits = self.hit_head(h)
        vel_pred = torch.sigmoid(self.vel_head(h))
        return hit_logits, vel_pred
