from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ChordModelConfig:
    feat_dim: int
    n_classes: int = 24          # 12 roots x {maj,min}
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    max_bars: int = 64           # must be >= seq_len used in preprocessing


class ChordTransformer(nn.Module):
    """
    Causal Transformer over bars.
    Input:  (B, T, F) bar-level features
    Output: (B, T, C) logits over chord classes
    """
    def __init__(self, cfg: ChordModelConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.feat_dim, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_bars, cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.n_classes)

        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns logits: (B, T, C)
        """
        B, T, F = x.shape
        if T > self.cfg.max_bars:
            raise ValueError(f"Sequence length T={T} exceeds cfg.max_bars={self.cfg.max_bars}")

        h = self.in_proj(x) + self.pos[:, :T, :]
        h = self.ln(h)

        # causal mask: prevent attending to future bars
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.tr(h, mask=mask)

        logits = self.out(h)
        return logits
