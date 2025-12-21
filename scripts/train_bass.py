# scripts/train_bass.py
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class BassModelConfig:
    feat_dim: int
    n_degree: int
    n_register: int
    n_rhythm: int
    max_steps: int = 128

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class BassTransformer(nn.Module):
    def __init__(self, cfg: BassModelConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.feat_dim, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_steps, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)

        self.head_degree = nn.Linear(cfg.d_model, cfg.n_degree)
        self.head_register = nn.Linear(cfg.d_model, cfg.n_register)
        self.head_rhythm = nn.Linear(cfg.d_model, cfg.n_rhythm)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # True means "masked" in PyTorch transformer
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,T,F)
        B, T, _ = x.shape
        device = x.device

        h = self.in_proj(x)
        idx = torch.arange(T, device=device)
        h = h + self.pos(idx)[None, :, :]

        causal = self._causal_mask(T, device=device)
        # src_key_padding_mask: True for PAD positions
        pad_mask = None
        if attn_mask is not None:
            pad_mask = ~attn_mask  # (B,T) True where padding

        h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.ln(h)

        deg = self.head_degree(h)
        reg = self.head_register(h)
        rhy = self.head_rhythm(h)
        return deg, reg, rhy


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    # y: (B,T), ignore_index masked
    mask = (y != IGNORE_INDEX)
    denom = int(mask.sum().item())
    if denom == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    correct = (pred == y) & mask
    return float(correct.sum().item() / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ml/bass_steps.npz")
    ap.add_argument("--out", type=str, default="data/ml/bass_model.pt")

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val_frac", type=float, default=0.1)

    # model knobs
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 => use NPZ seq_len")

    # loss weights
    ap.add_argument("--w_degree", type=float, default=1.0)
    ap.add_argument("--w_register", type=float, default=0.25)
    ap.add_argument("--w_rhythm", type=float, default=0.75)
    args = ap.parse_args()

    set_seed(int(args.seed))

    d = np.load(args.data)

    X = torch.tensor(d["X"], dtype=torch.float32)                 # (N,T,F)
    y_deg = torch.tensor(d["y_degree"], dtype=torch.int64)        # (N,T)
    y_reg = torch.tensor(d["y_register"], dtype=torch.int64)
    y_rhy = torch.tensor(d["y_rhythm"], dtype=torch.int64)

    attn = torch.tensor(d["attn_mask"], dtype=torch.bool)         # (N,T)
    label = torch.tensor(d["label_mask"], dtype=torch.bool)       # (N,T)

    # ignore positions where chord labels were unknown (preprocess marked these)
    valid = attn & label
    y_deg = y_deg.clone(); y_reg = y_reg.clone(); y_rhy = y_rhy.clone()
    y_deg[~valid] = IGNORE_INDEX
    y_reg[~valid] = IGNORE_INDEX
    y_rhy[~valid] = IGNORE_INDEX

    N, T, Fdim = X.shape

    n_degree = int(d["n_degree"][0]) if "n_degree" in d.files else int(y_deg[y_deg != IGNORE_INDEX].max().item() + 1)
    n_register = int(d["n_register"][0]) if "n_register" in d.files else 3
    n_rhythm = int(d["n_rhythm"][0]) if "n_rhythm" in d.files else int(y_rhy[y_rhy != IGNORE_INDEX].max().item() + 1)

    if int(args.max_steps) <= 0:
        args.max_steps = int(T)

    # train/val split
    idx = torch.randperm(N)
    n_val = int(float(args.val_frac) * N)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_ds = TensorDataset(X[tr_idx], y_deg[tr_idx], y_reg[tr_idx], y_rhy[tr_idx], attn[tr_idx])
    va_ds = TensorDataset(X[val_idx], y_deg[val_idx], y_reg[val_idx], y_rhy[val_idx], attn[val_idx])

    tr_dl = DataLoader(tr_ds, batch_size=int(args.batch), shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=int(args.batch), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = BassModelConfig(
        feat_dim=int(Fdim),
        n_degree=int(n_degree),
        n_register=int(n_register),
        n_rhythm=int(n_rhythm),
        max_steps=int(args.max_steps),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    )
    model = BassTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] X={tuple(X.shape)} seq_len={T} feat_dim={Fdim}")
    print(f"[classes] degree={n_degree} register={n_register} rhythm={n_rhythm}")
    print(f"[model] d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} max_steps={cfg.max_steps}")

    best_val = 1e18

    for epoch in range(1, int(args.epochs) + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_acc_d = tr_acc_r = tr_acc_h = 0.0
        n_batches = 0

        for xb, ydb, yrb, yhb, attb in tr_dl:
            xb = xb.to(device)
            ydb = ydb.to(device)
            yrb = yrb.to(device)
            yhb = yhb.to(device)
            attb = attb.to(device)

            deg_logits, reg_logits, rhy_logits = model(xb, attn_mask=attb)

            loss_d = F.cross_entropy(deg_logits.reshape(-1, cfg.n_degree), ydb.reshape(-1), ignore_index=IGNORE_INDEX)
            loss_r = F.cross_entropy(reg_logits.reshape(-1, cfg.n_register), yrb.reshape(-1), ignore_index=IGNORE_INDEX)
            loss_h = F.cross_entropy(rhy_logits.reshape(-1, cfg.n_rhythm), yhb.reshape(-1), ignore_index=IGNORE_INDEX)

            loss = float(args.w_degree) * loss_d + float(args.w_register) * loss_r + float(args.w_rhythm) * loss_h

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item())
            tr_acc_d += masked_accuracy(deg_logits, ydb)
            tr_acc_r += masked_accuracy(reg_logits, yrb)
            tr_acc_h += masked_accuracy(rhy_logits, yhb)
            n_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc_d /= max(1, n_batches)
        tr_acc_r /= max(1, n_batches)
        tr_acc_h /= max(1, n_batches)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_acc_d = va_acc_r = va_acc_h = 0.0
        n_batches = 0

        for xb, ydb, yrb, yhb, attb in va_dl:
            xb = xb.to(device)
            ydb = ydb.to(device)
            yrb = yrb.to(device)
            yhb = yhb.to(device)
            attb = attb.to(device)

            deg_logits, reg_logits, rhy_logits = model(xb, attn_mask=attb)

            loss_d = F.cross_entropy(deg_logits.reshape(-1, cfg.n_degree), ydb.reshape(-1), ignore_index=IGNORE_INDEX)
            loss_r = F.cross_entropy(reg_logits.reshape(-1, cfg.n_register), yrb.reshape(-1), ignore_index=IGNORE_INDEX)
            loss_h = F.cross_entropy(rhy_logits.reshape(-1, cfg.n_rhythm), yhb.reshape(-1), ignore_index=IGNORE_INDEX)

            loss = float(args.w_degree) * loss_d + float(args.w_register) * loss_r + float(args.w_rhythm) * loss_h

            va_loss += float(loss.item())
            va_acc_d += masked_accuracy(deg_logits, ydb)
            va_acc_r += masked_accuracy(reg_logits, yrb)
            va_acc_h += masked_accuracy(rhy_logits, yhb)
            n_batches += 1

        va_loss /= max(1, n_batches)
        va_acc_d /= max(1, n_batches)
        va_acc_r /= max(1, n_batches)
        va_acc_h /= max(1, n_batches)

        print(
            f"epoch {epoch:02d}  "
            f"train loss={tr_loss:.4f} acc(d/r/h)={tr_acc_d:.3f}/{tr_acc_r:.3f}/{tr_acc_h:.3f} | "
            f"val loss={va_loss:.4f} acc(d/r/h)={va_acc_d:.3f}/{va_acc_r:.3f}/{va_acc_h:.3f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "state": model.state_dict(),
                    "meta": {
                        "step_beats": float(d["step_beats"][0]) if "step_beats" in d.files else 2.0,
                        "include_key": bool(d["include_key"][0]) if "include_key" in d.files else False,
                    },
                },
                out_path,
            )
            print(f"  saved best -> {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
