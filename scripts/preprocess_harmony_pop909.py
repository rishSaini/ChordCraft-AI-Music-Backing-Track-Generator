from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi
from tqdm import tqdm

from backingtrack.midi_io import load_midi, extract_midi_info, build_bar_grid, pick_melody_instrument
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.types import Note
from backingtrack.key_detect import estimate_key


# --- Chord vocab (small but much richer than maj/min) ---
# Class 0 = N (no-chord)
QUALITIES = ("maj", "min", "7", "maj7", "min7", "dim", "sus2", "sus4")
N_CLASS = 0

NOTE2PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _parse_root_pc(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    letter = s[0].upper()
    if letter not in NOTE2PC:
        return None
    pc = NOTE2PC[letter]
    if len(s) >= 2:
        if s[1] == "#":
            pc = (pc + 1) % 12
        elif s[1].lower() == "b":
            pc = (pc - 1) % 12
    return int(pc)


def _normalize_quality(q: str) -> Optional[str]:
    q = (q or "").strip().lower()
    if q in ("", "maj", "major"):
        return "maj"
    if q in ("min", "minor", "m"):
        return "min"

    # Common pop chord spellings
    if q in ("7", "dom7", "dominant7", "9", "11", "13"):
        return "7"
    if q in ("maj7", "major7", "maj9", "maj11", "maj13"):
        return "maj7"
    if q in ("min7", "m7", "minor7", "min9", "min11", "min13"):
        return "min7"

    if "sus2" in q:
        return "sus2"
    if "sus4" in q or q == "sus":
        return "sus4"

    if "dim" in q or "hdim" in q:
        return "dim"

    # If we can't map it, ignore this label
    return None


def chord_to_id(chord_name: str) -> Tuple[int, bool]:
    """
    Returns (class_id, is_known).
    POP909 chord names are typically like:
      C:maj, A:min, G:7, N
    """
    name = chord_name.strip()
    if not name or name.upper() == "N":
        return N_CLASS, True

    # split root and quality
    if ":" in name:
        root_s, qual_s = name.split(":", 1)
    else:
        # sometimes formats like "C" appear -> treat as maj
        root_s, qual_s = name, "maj"

    root_pc = _parse_root_pc(root_s)
    qual = _normalize_quality(qual_s)

    if root_pc is None or qual is None:
        return -100, False

    q_idx = QUALITIES.index(qual)
    cid = 1 + q_idx * 12 + root_pc
    return int(cid), True


def id_to_chord(cid: int) -> str:
    if cid == 0:
        return "N"
    cid = int(cid) - 1
    q_idx = cid // 12
    root = cid % 12
    pc_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{pc_names[root]}:{QUALITIES[q_idx]}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


@dataclass(frozen=True)
class ChordSeg:
    start: float
    end: float
    cid: int
    known: bool


def load_pop909_chords(song_dir: Path) -> List[ChordSeg]:
    # Prefer chord_midi.txt, fall back to chord_audio.txt if needed
    cand = [song_dir / "chord_midi.txt", song_dir / "chord_audio.txt"]
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        return []

    segs: List[ChordSeg] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            s = float(parts[0])
            e = float(parts[1])
        except ValueError:
            continue
        name = parts[2]
        cid, known = chord_to_id(name)
        if e <= s:
            continue
        segs.append(ChordSeg(start=s, end=e, cid=cid, known=known))
    segs.sort(key=lambda x: x.start)
    return segs


def _key_features(melody_notes: List[Note]) -> np.ndarray:
    k = estimate_key(melody_notes)
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0)


def extract_step_features(
    melody: List[Note],
    step_start: float,
    step_end: float,
    step_len: float,
    key_feat: Optional[np.ndarray],
) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float32)
    tot = 0.0
    pitch_num = 0.0

    for n in melody:
        ov = _overlap(n.start, n.end, step_start, step_end)
        if ov <= 0:
            continue
        tot += ov
        hist[int(n.pitch) % 12] += float(ov)
        pitch_num += float(ov) * float(n.pitch)

    if hist.sum() > 1e-9:
        hist = hist / (hist.sum() + 1e-9)

    mean_pitch = (pitch_num / max(1e-9, tot)) if tot > 0 else 60.0
    mean_pitch_norm = float(np.clip(mean_pitch / 127.0, 0.0, 1.0))
    activity = float(np.clip(tot / max(1e-9, step_len), 0.0, 1.0))

    parts = [hist, np.array([mean_pitch_norm, activity], dtype=np.float32)]
    if key_feat is not None:
        parts.append(key_feat.astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def label_step(segs: List[ChordSeg], t0: float, t1: float) -> Tuple[int, bool]:
    """
    Pick chord label by max-overlap with this step.
    If none overlaps, return N.
    """
    best = None
    best_ov = 0.0
    for s in segs:
        ov = _overlap(s.start, s.end, t0, t1)
        if ov > best_ov:
            best_ov = ov
            best = s
    if best is None or best_ov <= 0.0:
        return N_CLASS, True
    return int(best.cid), bool(best.known)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909_root", type=str, required=True, help="Path to POP909 root with 001/002/... folders")
    ap.add_argument("--out", type=str, default="data/ml/chords_steps.npz")
    ap.add_argument("--only_4_4", action="store_true")
    ap.add_argument("--include_key", action="store_true")
    ap.add_argument("--seq_len", type=int, default=128, help="sequence length in steps (NOT bars)")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--step_beats", type=float, default=2.0, help="2.0 => half-bar in 4/4")
    args = ap.parse_args()

    root = Path(args.pop909_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_windows = []
    y_windows = []
    attn_windows = []
    label_windows = []

    song_dirs = [p for p in root.iterdir() if p.is_dir()]
    song_dirs.sort(key=lambda p: p.name)

    melody_cfg = MelodyConfig(quantize_to_beat=False)

    for sd in tqdm(song_dirs, desc="POP909 songs"):
        # use main midi: <idx>/<idx>.mid
        midi_path = sd / f"{sd.name}.mid"
        if not midi_path.exists():
            continue

        try:
            pm = load_midi(midi_path)
            info = extract_midi_info(pm)
        except Exception:
            continue

        if args.only_4_4:
            if not (info.time_signature.numerator == 4 and info.time_signature.denominator == 4):
                continue

        grid = build_bar_grid(info)

        # Melody track preference (POP909 has MELODY)
        mel_inst = None
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            if "melody" in (inst.name or "").lower():
                mel_inst = inst
                break
        if mel_inst is None:
            mel_inst, _ = pick_melody_instrument(pm)

        melody = extract_melody_notes(mel_inst, grid=None, config=melody_cfg)
        if not melody:
            continue

        segs = load_pop909_chords(sd)
        if not segs:
            continue

        key_feat = _key_features(melody) if args.include_key else None

        step_len = float(args.step_beats) * float(grid.seconds_per_beat)
        if step_len <= 0:
            continue

        dur = float(info.duration)
        n_steps = int(np.ceil(max(1e-6, dur) / step_len))

        # build per-step arrays
        feats = []
        labels = []
        known_mask = []

        for i in range(n_steps):
            t0 = i * step_len
            t1 = t0 + step_len
            feats.append(extract_step_features(melody, t0, t1, step_len, key_feat))

            cid, known = label_step(segs, t0, t1)
            labels.append(int(cid) if known else -100)
            known_mask.append(bool(known))

        X = np.stack(feats, axis=0).astype(np.float32)          # (T,F)
        y = np.array(labels, dtype=np.int64)                    # (T,)
        known_mask = np.array(known_mask, dtype=np.bool_)       # (T,)

        # window into fixed length sequences
        seq_len = int(args.seq_len)
        stride = int(args.stride)

        for start in range(0, n_steps, stride):
            end = start + seq_len
            xw = np.zeros((seq_len, X.shape[1]), dtype=np.float32)
            yw = np.full((seq_len,), -100, dtype=np.int64)
            attn = np.zeros((seq_len,), dtype=np.bool_)
            labm = np.zeros((seq_len,), dtype=np.bool_)

            sl = slice(start, min(end, n_steps))
            L = sl.stop - sl.start
            if L <= 0:
                continue

            xw[:L] = X[sl]
            yw[:L] = y[sl]
            attn[:L] = True
            labm[:L] = known_mask[sl]  # only train on recognized labels

            X_windows.append(xw)
            y_windows.append(yw)
            attn_windows.append(attn)
            label_windows.append(labm)

    if not X_windows:
        raise RuntimeError("No training windows were created. Check paths + that chord_midi.txt exists per song.")

    X_out = np.stack(X_windows, axis=0)
    y_out = np.stack(y_windows, axis=0)
    attn_out = np.stack(attn_windows, axis=0)
    lab_out = np.stack(label_windows, axis=0)

    np.savez_compressed(
        out_path,
        X=X_out,
        y=y_out,
        attn_mask=attn_out,
        label_mask=lab_out,
        qualities=np.array(QUALITIES, dtype=object),
        n_classes=np.array([1 + 12 * len(QUALITIES)], dtype=np.int64),
        step_beats=np.array([float(args.step_beats)], dtype=np.float32),
    )
    print(f"Saved: {out_path}")
    print(f"Windows: {len(X_out)} | feat_dim={X_out.shape[-1]} | n_classes={1 + 12*len(QUALITIES)} | step_beats={args.step_beats}")


if __name__ == "__main__":
    main()
