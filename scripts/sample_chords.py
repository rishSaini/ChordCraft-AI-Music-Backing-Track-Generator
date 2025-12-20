from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi
import torch

from backingtrack.key_detect import estimate_key
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.midi_io import build_bar_grid, extract_midi_info, load_midi, pick_melody_instrument
from backingtrack.types import Note
from backingtrack.ml_harmony.model import ChordModelConfig, ChordTransformer

PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def chord_name(cid: int) -> str:
    root = int(cid % 12)
    qual = "maj" if int(cid // 12) == 0 else "min"
    return f"{PC_NAMES[root]}:{qual}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def _bucket_notes_by_bar(notes: List[pretty_midi.Note], bar_len: float, n_bars: int) -> List[List[pretty_midi.Note]]:
    buckets: List[List[pretty_midi.Note]] = [[] for _ in range(n_bars)]
    eps = 1e-6
    for n in notes:
        if n.end <= n.start:
            continue
        b0 = int(max(0.0, n.start) // bar_len)
        b1 = int(max(0.0, (n.end - eps)) // bar_len)
        if b0 >= n_bars:
            continue
        b1 = min(b1, n_bars - 1)
        for b in range(max(0, b0), b1 + 1):
            buckets[b].append(n)
    return buckets


def _mel_feats_bar(mel_notes: List[pretty_midi.Note], bs: float, be: float, bar_len: float) -> Tuple[np.ndarray, float, float]:
    hist = np.zeros(12, dtype=np.float32)
    tot = 0.0
    pitch_num = 0.0
    for n in mel_notes:
        ov = _overlap(float(n.start), float(n.end), bs, be)
        if ov <= 0.0:
            continue
        tot += ov
        hist[int(n.pitch) % 12] += float(ov)
        pitch_num += float(ov) * float(n.pitch)
    if hist.sum() > 1e-9:
        hist = hist / (hist.sum() + 1e-9)
    mean_pitch = (pitch_num / max(1e-9, tot)) if tot > 0 else 60.0
    mean_pitch_norm = float(np.clip(mean_pitch / 127.0, 0.0, 1.0))
    activity = float(np.clip(tot / max(1e-9, bar_len), 0.0, 1.0))
    return hist, mean_pitch_norm, activity


def _key_features(melody_notes: List[Note]) -> np.ndarray:
    k = estimate_key(melody_notes)
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0)


def load_model(path: str | Path) -> Tuple[ChordTransformer, ChordModelConfig, torch.device]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(path), map_location=dev)
    cfg = ChordModelConfig(**ckpt["cfg"])
    model = ChordTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg, dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", type=str, required=True)
    ap.add_argument("--model", type=str, default="data/ml/chord_model.pt")
    ap.add_argument("--melody_index", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--include_key", action="store_true")
    args = ap.parse_args()

    pm = load_midi(args.midi)
    info = extract_midi_info(pm)
    grid = build_bar_grid(info)

    if not (info.time_signature.numerator == 4 and info.time_signature.denominator == 4):
        raise ValueError("Chord model v1 expects 4/4 (same as preprocessing).")

    bar_len = float(grid.bar_duration)
    dur = float(pm.get_end_time())
    n_bars = max(1, grid.bar_index_at(dur - 1e-6) + 1)

    mel_inst, _ = pick_melody_instrument(pm, instrument_index=args.melody_index)
    melody_cfg = MelodyConfig(
        min_note_duration=0.05,
        overlap_tolerance=0.02,
        merge_gap=0.03,
        strategy="highest_pitch",
        quantize_to_beat=False,
    )
    mel_line = extract_melody_notes(mel_inst, grid=None, config=melody_cfg)
    mel_pm = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in mel_line]

    mel_by_bar = _bucket_notes_by_bar(mel_pm, bar_len=bar_len, n_bars=n_bars)

    key_feat = _key_features([Note(pitch=n.pitch, start=n.start, end=n.end, velocity=n.velocity) for n in mel_pm]) if args.include_key else None

    feats = []
    for b in range(n_bars):
        bs = b * bar_len
        be = bs + bar_len
        hist, mp, act = _mel_feats_bar(mel_by_bar[b], bs, be, bar_len)
        parts = [hist.astype(np.float32), np.array([mp, act], dtype=np.float32)]
        if key_feat is not None:
            parts.append(key_feat.astype(np.float32))
        feats.append(np.concatenate(parts, axis=0).astype(np.float32))

    X = np.stack(feats, axis=0)  # (B,F)

    model, cfg, dev = load_model(args.model)

    # chunk into seq_len windows
    print(f"Bars: {n_bars}")
    for start in range(0, n_bars, int(args.seq_len)):
        end = min(n_bars, start + int(args.seq_len))
        chunk = X[start:end]
        # pad to seq_len
        if chunk.shape[0] < int(args.seq_len):
            pad = np.zeros((int(args.seq_len) - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        xb = torch.tensor(chunk[None, :, :], dtype=torch.float32, device=dev)
        logits = model(xb)[0]  # (T,C)
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()

        for i in range(start, end):
            print(f"bar {i:03d}: {chord_name(pred[i - start])}")


if __name__ == "__main__":
    main()
