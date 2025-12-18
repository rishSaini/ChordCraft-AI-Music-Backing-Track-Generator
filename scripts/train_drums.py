from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backingtrack.ml_drums.model import DrumModelConfig, DrumTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ml/drum_bars.npz")
    ap.add_argument("--out", type=str, default="data/ml/drum_model.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    d = np.load(args.data)
    hits = torch.tensor(d["hits"], dtype=torch.float32)  # (N,16,V)
    vels = torch.tensor(d["vels"], dtype=torch.float32)  # (N,16,V)

    # Teacher forcing input: shift right by 1 step (start token = zeros)
    zeros = torch.zeros((hits.shape[0], 1, hits.shape[2]), dtype=torch.float32)
    x_hits = torch.cat([zeros, hits[:, :-1, :]], dim=1)
    x_vels = torch.cat([zeros, vels[:, :-1, :]], dim=1)
    x = torch.cat([x_hits, x_vels], dim=2)  # (N,16,2V)

    ds = TensorDataset(x, hits, vels)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DrumModelConfig(n_voices=hits.shape[2])
    model = DrumTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # imbalance handling for hits
    pos = hits.mean(dim=(0, 1)).clamp(1e-4, 1 - 1e-4)  # per-voice hit rate
    pos_weight = ((1 - pos) / pos).to(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0

        for xb, y_hits, y_vels in dl:
            xb = xb.to(device)
            y_hits = y_hits.to(device)
            y_vels = y_vels.to(device)

            hit_logits, vel_pred = model(xb)

            # hit loss
            hit_loss = F.binary_cross_entropy_with_logits(hit_logits, y_hits, pos_weight=pos_weight)

            # velocity loss only where hit exists
            mask = (y_hits > 0.5).float()
            vel_loss = ((vel_pred - y_vels) ** 2 * mask).sum() / (mask.sum().clamp_min(1.0))

            loss = hit_loss + 0.25 * vel_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())

        avg = total / max(1, len(dl))
        print(f"epoch {epoch:02d}  loss={avg:.4f}")

    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    main()
