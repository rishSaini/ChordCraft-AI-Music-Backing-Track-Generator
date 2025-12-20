from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backingtrack.ml_harmony.model import ChordModelConfig, ChordTransformer


IGNORE_INDEX = -100


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    logits: (B,T,C)
    y: (B,T)
    valid_mask: (B,T) bool
    """
    pred = logits.argmax(dim=-1)
    correct = (pred == y) & valid_mask
    denom = valid_mask.sum().item()
    return float(correct.sum().item() / max(1, denom))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ml/chords_seq.npz")
    ap.add_argument("--out", type=str, default="data/ml/chord_model.pt")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=64, help="must match preprocessing seq_len")
    ap.add_argument("--max_bars", type=int, default=64, help=">= seq_len")

    args = ap.parse_args()
    _set_seed(int(args.seed))

    d = np.load(args.data)
    X = torch.tensor(d["X"], dtype=torch.float32)            # (N,T,F)
    y = torch.tensor(d["y"], dtype=torch.int64)              # (N,T)
    attn = torch.tensor(d["attn_mask"], dtype=torch.bool)    # (N,T)
    label = torch.tensor(d["label_mask"], dtype=torch.bool)  # (N,T)

    if X.shape[1] != int(args.seq_len):
        raise ValueError(f"NPZ seq_len={X.shape[1]} but args --seq_len={args.seq_len}")

    # Only train on positions that are real bars AND deemed reliable by preprocessing
    valid = attn & label & (y != IGNORE_INDEX)

    # Set targets to IGNORE where invalid so we can use ignore_index
    y_train = y.clone()
    y_train[~valid] = IGNORE_INDEX

    # split train/val
    N = X.shape[0]
    idx = torch.randperm(N)
    n_val = int(N * float(args.val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_ds = TensorDataset(X[tr_idx], y_train[tr_idx], valid[tr_idx])
    va_ds = TensorDataset(X[val_idx], y_train[val_idx], valid[val_idx])

    tr_dl = DataLoader(tr_ds, batch_size=int(args.batch), shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=int(args.batch), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(np.load(args.data)["n_classes"][0])
    
    cfg = ChordModelConfig(
        feat_dim=int(X.shape[2]),
        n_classes=n_classes,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        max_bars=int(args.max_bars),
    )
    model = ChordTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = 1e18
    for epoch in range(1, int(args.epochs) + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n_batches = 0

        for xb, yb, vb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)

            logits = model(xb)  # (B,T,24)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item())
            tr_acc += _accuracy(logits, yb, vb)
            n_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc /= max(1, n_batches)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        n_batches = 0

        for xb, yb, vb in va_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            va_loss += float(loss.item())
            va_acc += _accuracy(logits, yb, vb)
            n_batches += 1

        va_loss /= max(1, n_batches)
        va_acc /= max(1, n_batches)

        print(f"epoch {epoch:02d}  train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")

        # save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, out_path)
            print(f"  saved best -> {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
