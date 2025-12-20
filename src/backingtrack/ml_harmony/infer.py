# src/backingtrack/ml_harmony/infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..types import BarGrid, Note
from ..key_detect import estimate_key
from ..harmony_baseline import ChordEvent
from .model import ChordModelConfig, ChordTransformer

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _decode_chord_id(cid: int) -> tuple[int, str]:
    """
    Your v1 chord vocab:
      0..11   -> root 0..11, maj
      12..23  -> root 0..11, min
    """
    cid = int(cid)
    root_pc = cid % 12
    quality = "maj" if (cid // 12) == 0 else "min"
    return int(root_pc), quality


def chord_name(cid: int) -> str:
    root, qual = _decode_chord_id(cid)
    return f"{PC_NAMES[root]}:{qual}"


def load_chord_model(model_path: str | Path, device: Optional[str] = None) -> tuple[ChordTransformer, ChordModelConfig, torch.device]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Chord model not found: {p}")

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    cfg = ChordModelConfig(**ckpt["cfg"])
    model = ChordTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg, dev


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def _key_features(melody_notes: List[Note]) -> np.ndarray:
    """
    14 dims: tonic onehot(12) + mode onehot(2)
    Must match preprocess_harmony.py.
    """
    k = estimate_key(melody_notes)
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0)


def extract_bar_features_from_melody(
    melody_notes: List[Note],
    grid: BarGrid,
    *,
    num_bars: int,
    include_key: bool,
) -> np.ndarray:
    """
    Matches the features used in preprocess_harmony.py:
      - melody pitch-class histogram (12) normalized by total overlap duration
      - mean_pitch_norm (1) duration-weighted
      - activity_ratio (1) total overlap / bar_len
      - optional key features (14) repeated each bar
    Returns X: (B, F) float32
    """
    bar_len = float(grid.bar_duration)
    spb = float(grid.seconds_per_beat)  # not used directly but kept for consistency

    key_feat = _key_features(melody_notes).astype(np.float32) if include_key else None

    X = []
    for b in range(int(num_bars)):
        bs = float(grid.time_at(b, 0.0))
        be = float(grid.time_at(b + 1, 0.0))

        hist = np.zeros(12, dtype=np.float32)
        tot = 0.0
        pitch_num = 0.0

        for n in melody_notes:
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

        parts = [hist.astype(np.float32), np.array([mean_pitch_norm, activity], dtype=np.float32)]
        if key_feat is not None:
            parts.append(key_feat)

        X.append(np.concatenate(parts, axis=0).astype(np.float32))

    return np.stack(X, axis=0).astype(np.float32)


def _viterbi_smooth(logits: np.ndarray, change_penalty: float) -> List[int]:
    """
    Smooth chord sequence by penalizing changes between adjacent bars.
    logits: (T, C)
    returns list[int] length T
    """
    if change_penalty <= 0:
        return logits.argmax(axis=-1).astype(int).tolist()

    # log-probabilities for numerical stability
    x = torch.tensor(logits, dtype=torch.float32)
    logp = torch.log_softmax(x, dim=-1).cpu().numpy()  # (T,C)

    T, C = logp.shape
    dp = np.full((T, C), 1e18, dtype=np.float64)
    bp = np.zeros((T, C), dtype=np.int32)

    # cost = -logp
    dp[0] = -logp[0]

    for t in range(1, T):
        for c in range(C):
            # min over previous states with change penalty
            costs = dp[t - 1] + (0.0 if False else 0.0)  # placeholder
            # vectorize: add penalty where prev != c
            penalty = np.full((C,), float(change_penalty), dtype=np.float64)
            penalty[c] = 0.0
            costs = dp[t - 1] + penalty
            j = int(np.argmin(costs))
            dp[t, c] = float(costs[j]) + float(-logp[t, c])
            bp[t, c] = j

    last = int(np.argmin(dp[T - 1]))
    path = [0] * T
    path[T - 1] = last
    for t in range(T - 2, -1, -1):
        path[t] = int(bp[t + 1, path[t + 1]])
    return path


def predict_chord_ids(
    melody_notes: List[Note],
    grid: BarGrid,
    *,
    duration_seconds: float,
    model_path: str = "data/ml/chord_model.pt",
    include_key: bool = True,
    change_penalty: float = 0.6,
    device: Optional[str] = None,
) -> List[int]:
    """
    Returns one chord_id per bar for the whole song.
    """
    model, cfg, dev = load_chord_model(model_path, device=device)

    num_bars = int(grid.bar_index_at(max(0.0, float(duration_seconds) - 1e-6)) + 1)
    X = extract_bar_features_from_melody(melody_notes, grid, num_bars=num_bars, include_key=include_key)

    if X.shape[1] != int(cfg.feat_dim):
        # common mismatch: trained with/without key features
        raise ValueError(
            f"Feature dim mismatch. Model expects feat_dim={cfg.feat_dim}, "
            f"but extracted features have dim={X.shape[1]}. "
            f"Did you train with --include_key but infer with include_key={include_key}?"
        )

    T = X.shape[0]
    max_bars = int(cfg.max_bars)

    # If song fits in one forward pass
    if T <= max_bars:
        xb = torch.tensor(X[None, :, :], dtype=torch.float32, device=dev)
        with torch.no_grad():
            logits = model(xb)[0].cpu().numpy()  # (T,C)
        return _viterbi_smooth(logits, change_penalty=float(change_penalty))

    # Sliding window fallback for very long songs
    out_ids: List[int] = []
    start = 0
    while start < T:
        end = min(T, start + max_bars)
        chunk = X[start:end]
        if chunk.shape[0] < max_bars:
            pad = np.zeros((max_bars - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        xb = torch.tensor(chunk[None, :, :], dtype=torch.float32, device=dev)
        with torch.no_grad():
            logits = model(xb)[0].cpu().numpy()  # (max_bars,C)

        keep = end - start
        ids = _viterbi_smooth(logits[:keep], change_penalty=float(change_penalty))
        out_ids.extend(ids)

        start = end

    return out_ids


def chord_events_from_ids(
    chord_ids: List[int],
    grid: BarGrid,
) -> List[ChordEvent]:
    """
    Convert per-bar chord ids into merged ChordEvent spans.
    """
    events: List[ChordEvent] = []
    if not chord_ids:
        return events

    def make_event(bar0: int, bar1_excl: int, cid: int) -> ChordEvent:
        root, qual = _decode_chord_id(cid)
        start = float(grid.time_at(bar0, 0.0))
        end = float(grid.time_at(bar1_excl, 0.0))
        return ChordEvent(root_pc=int(root), quality=str(qual), extensions=tuple(), bar_index=int(bar0), start=start, end=end)

    run_start = 0
    prev = chord_ids[0]
    for i in range(1, len(chord_ids)):
        if chord_ids[i] != prev:
            events.append(make_event(run_start, i, prev))
            run_start = i
            prev = chord_ids[i]
    events.append(make_event(run_start, len(chord_ids), prev))
    return events


def predict_chords_ml(
    melody_notes: List[Note],
    grid: BarGrid,
    *,
    duration_seconds: float,
    model_path: str = "data/ml/chord_model.pt",
    include_key: bool = True,
    change_penalty: float = 0.6,
    device: Optional[str] = None,
) -> List[ChordEvent]:
    ids = predict_chord_ids(
        melody_notes,
        grid,
        duration_seconds=duration_seconds,
        model_path=model_path,
        include_key=include_key,
        change_penalty=change_penalty,
        device=device,
    )
    return chord_events_from_ids(ids, grid)
