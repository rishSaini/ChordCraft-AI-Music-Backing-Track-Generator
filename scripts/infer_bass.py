# scripts/infer_bass.py
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi
import torch
import torch.nn as nn

from backingtrack.midi_io import load_and_prepare
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.key_detect import estimate_key
from backingtrack.types import BarGrid, Note

# -----------------------------
# Chord predictor loader (robust)
# -----------------------------
def _load_chord_predictor():
    """
    Prefer step-based chord predictor if present, else fall back to bar-based ML chords.
    Returns (callable, source_name).
    """
    try:
        from backingtrack.ml_harmony.steps_infer import predict_chords_steps as fn  # type: ignore

        return fn, "backingtrack.ml_harmony.steps_infer.predict_chords_steps"
    except Exception:
        from backingtrack.ml_harmony.infer import predict_chords_ml as fn  # type: ignore

        return fn, "backingtrack.ml_harmony.infer.predict_chords_ml"


def _call_chord_predictor(
    fn,
    *,
    melody_notes: List[Note],
    grid: BarGrid,
    duration_seconds: float,
    model_path: str,
    include_key: bool,
    step_beats: float,
) -> List[Any]:
    """
    Calls chord predictor using only supported kwargs.
    Fixes errors like: unexpected keyword argument 'step_beats'
    """
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}

    if "melody_notes" in sig.parameters:
        kwargs["melody_notes"] = melody_notes
    if "grid" in sig.parameters:
        kwargs["grid"] = grid
    if "duration_seconds" in sig.parameters:
        kwargs["duration_seconds"] = float(duration_seconds)
    elif "duration" in sig.parameters:
        kwargs["duration"] = float(duration_seconds)

    # model path naming differences
    if "model_path" in sig.parameters:
        kwargs["model_path"] = model_path
    elif "model" in sig.parameters:
        kwargs["model"] = model_path
    elif "checkpoint" in sig.parameters:
        kwargs["checkpoint"] = model_path

    if "include_key" in sig.parameters:
        kwargs["include_key"] = bool(include_key)

    if "step_beats" in sig.parameters:
        kwargs["step_beats"] = float(step_beats)

    out = fn(**kwargs)
    return list(out) if out is not None else []


# -----------------------------
# Bass model (must match train_bass.py)
# -----------------------------
IGNORE_INDEX = -100


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

def _normalize_bass_cfg(cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes older/newer cfg dictionaries compatible with inference code.

    Handles:
      - n_degrees -> n_degree
      - n_registers -> n_register
      - n_rhythms -> n_rhythm
      - max_len/seq_len -> max_steps
    Also filters out any unknown keys.
    """
    cfg = dict(cfg_in)

    # Aliases (either direction)
    if "n_degrees" in cfg and "n_degree" not in cfg:
        cfg["n_degree"] = cfg.pop("n_degrees")
    if "n_registers" in cfg and "n_register" not in cfg:
        cfg["n_register"] = cfg.pop("n_registers")
    if "n_rhythms" in cfg and "n_rhythm" not in cfg:
        cfg["n_rhythm"] = cfg.pop("n_rhythms")

    if "max_len" in cfg and "max_steps" not in cfg:
        cfg["max_steps"] = cfg.pop("max_len")
    if "seq_len" in cfg and "max_steps" not in cfg:
        cfg["max_steps"] = cfg.pop("seq_len")

    allowed = {f.name for f in fields(BassModelConfig)}
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    return cfg


class BassTransformer(nn.Module):
    """
    Must match scripts/train_bass.py architecture:
      - in_proj
      - pos embedding
      - TransformerEncoder with causal mask
      - 3 heads: degree/register/rhythm
    """

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
        # True means masked
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x: (B,T,F)
        B, T, _ = x.shape
        device = x.device

        h = self.in_proj(x)
        idx = torch.arange(T, device=device)
        h = h + self.pos(idx)[None, :, :]

        causal = self._causal_mask(T, device=device)

        pad_mask = None
        if attn_mask is not None:
            # attn_mask True where valid, but transformer expects True where padding
            pad_mask = ~attn_mask

        h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.ln(h)

        deg = self.head_degree(h)
        reg = self.head_register(h)
        rhy = self.head_rhythm(h)
        return deg, reg, rhy


def _normalize_bass_cfg(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make older/newer cfg dicts compatible.
    Accepts n_degree/n_degrees, etc.
    """
    cfg = dict(cfg_dict)

    # common renames (just in case)
    if "n_degrees" in cfg and "n_degree" not in cfg:
        cfg["n_degree"] = cfg.pop("n_degrees")
    if "n_registers" in cfg and "n_register" not in cfg:
        cfg["n_register"] = cfg.pop("n_registers")
    if "n_rhythms" in cfg and "n_rhythm" not in cfg:
        cfg["n_rhythm"] = cfg.pop("n_rhythms")

    # sometimes called max_len/seq_len
    if "max_len" in cfg and "max_steps" not in cfg:
        cfg["max_steps"] = cfg.pop("max_len")
    if "seq_len" in cfg and "max_steps" not in cfg:
        cfg["max_steps"] = cfg.pop("seq_len")

    return cfg


def load_bass_model(path: str | Path, device: Optional[str] = None) -> Tuple[BassTransformer, BassModelConfig, torch.device, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bass model not found: {p}")

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    ckpt = torch.load(str(p), map_location=dev)
    cfg_dict = _normalize_bass_cfg(ckpt["cfg"])
    cfg = BassModelConfig(**cfg_dict)

    model = BassTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    meta = ckpt.get("meta", {})
    return model, cfg, dev, meta


# -----------------------------
# Features must match preprocess_bass.py
# -----------------------------
QUAL_VOCAB = ["N", "maj", "min", "7", "maj7", "min7", "dim", "sus2", "sus4"]
QUAL_TO_I: Dict[str, int] = {q: i for i, q in enumerate(QUAL_VOCAB)}

# degree labels (from preprocess_bass.py)
DEG_VOCAB = ["REST", "ROOT", "THIRD", "FIFTH", "SEVENTH", "CHORD_TONE", "NONCHORD"]
RHY_VOCAB = ["REST", "SUSTAIN", "HIT_ON", "HIT_OFF", "MULTI"]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def key_features(melody: Sequence[Note]) -> np.ndarray:
    k = estimate_key(list(melody))
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0).astype(np.float32)


def melody_step_features(melody: Sequence[Note], t0: float, t1: float, step_len: float) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float32)
    tot = 0.0
    pitch_num = 0.0
    for n in melody:
        ov = _overlap(float(n.start), float(n.end), t0, t1)
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

    return np.concatenate([hist, np.array([mean_pitch_norm, activity], dtype=np.float32)], axis=0).astype(np.float32)


@dataclass(frozen=True)
class ChordEventLite:
    start: float
    end: float
    root_pc: Optional[int]  # None for N
    quality: str            # one of QUAL_VOCAB or "N"


def _normalize_quality(q: str) -> str:
    q = (q or "").strip().lower()
    if q in ("n", "none", "no_chord", "nochord"):
        return "N"
    # keep within our vocab
    if q in QUAL_TO_I:
        return q
    # map some common variants
    if q in ("major",):
        return "maj"
    if q in ("minor", "m"):
        return "min"
    if q in ("dom7", "dominant7", "9", "11", "13"):
        return "7"
    if q in ("major7", "maj9"):
        return "maj7"
    if q in ("minor7", "m7", "min9"):
        return "min7"
    if "sus2" in q:
        return "sus2"
    if "sus4" in q or q == "sus":
        return "sus4"
    if "dim" in q:
        return "dim"
    # fallback
    return "maj"


def chord_step_features(ch: ChordEventLite) -> np.ndarray:
    root_oh = np.zeros(12, dtype=np.float32)
    if ch.root_pc is not None:
        root_oh[int(ch.root_pc) % 12] = 1.0

    qual_oh = np.zeros(len(QUAL_VOCAB), dtype=np.float32)
    q = _normalize_quality(ch.quality)
    qual_oh[QUAL_TO_I.get(q, 0)] = 1.0  # default to "N"

    return np.concatenate([root_oh, qual_oh], axis=0).astype(np.float32)


def step_in_bar_onehot(t0: float, spb: float, step_beats: float) -> np.ndarray:
    bins = 2 if step_beats >= 2.0 else 4
    oh = np.zeros(bins, dtype=np.float32)
    beats = (t0 / max(1e-9, spb)) % 4.0
    idx = int(np.floor(beats / (4.0 / bins))) % bins
    oh[idx] = 1.0
    return oh


def chord_at(chords: Sequence[ChordEventLite], t: float) -> ChordEventLite:
    for c in chords:
        if c.start <= t < c.end:
            return c
    return chords[-1]


def get_chords_from_ml(
    *,
    melody_notes: List[Note],
    grid: BarGrid,
    duration_seconds: float,
    chord_model_path: str,
    include_key: bool,
    step_beats: float,
) -> List[ChordEventLite]:
    fn, src = _load_chord_predictor()
    evs = _call_chord_predictor(
        fn,
        melody_notes=melody_notes,
        grid=grid,
        duration_seconds=duration_seconds,
        model_path=chord_model_path,
        include_key=include_key,
        step_beats=step_beats,
    )

    out: List[ChordEventLite] = []
    for e in evs:
        start = float(getattr(e, "start", 0.0))
        end = float(getattr(e, "end", 0.0))
        root = getattr(e, "root_pc", None)
        qual = getattr(e, "quality", "N")

        qn = _normalize_quality(str(qual))
        if qn == "N":
            out.append(ChordEventLite(start=start, end=end, root_pc=None, quality="N"))
        else:
            out.append(ChordEventLite(start=start, end=end, root_pc=int(root) % 12 if root is not None else 0, quality=qn))

    out.sort(key=lambda x: x.start)
    if not out:
        out = [ChordEventLite(0.0, float(duration_seconds), None, "N")]
    return out


# -----------------------------
# Render bass from predictions (degree/register/rhythm)
# -----------------------------
def chord_pitch_classes(root_pc: Optional[int], quality: str) -> Tuple[int, ...]:
    if root_pc is None or quality == "N":
        return tuple()
    q = _normalize_quality(quality)
    r = int(root_pc) % 12

    if q == "maj":
        ivs = (0, 4, 7)
    elif q == "min":
        ivs = (0, 3, 7)
    elif q == "dim":
        ivs = (0, 3, 6)
    elif q == "sus2":
        ivs = (0, 2, 7)
    elif q == "sus4":
        ivs = (0, 5, 7)
    elif q == "7":
        ivs = (0, 4, 7, 10)
    elif q == "maj7":
        ivs = (0, 4, 7, 11)
    elif q == "min7":
        ivs = (0, 3, 7, 10)
    else:
        ivs = (0, 4, 7)

    pcs = [(r + iv) % 12 for iv in ivs]
    out: List[int] = []
    for pc in pcs:
        if pc not in out:
            out.append(int(pc))
    return tuple(out)


def pc_to_pitch(pc: int, register_bin: int) -> int:
    centers = [40, 50, 60]  # low/mid/high centers
    center = centers[int(np.clip(register_bin, 0, 2))]
    best = None
    best_dist = 1e9
    for pitch in range(24, 84):
        if pitch % 12 != (pc % 12):
            continue
        dist = abs(pitch - center)
        if dist < best_dist:
            best_dist = dist
            best = pitch
    return int(best if best is not None else center)


def render_bass_step(
    t0: float,
    step_len: float,
    degree_id: int,
    register_id: int,
    rhythm_id: int,
    chord: ChordEventLite,
    velocity: int,
) -> List[Note]:
    # degree_id uses DEG_VOCAB indexing:
    # 0 REST, 1 ROOT, 2 THIRD, 3 FIFTH, 4 SEVENTH, 5 CHORD_TONE, 6 NONCHORD
    if rhythm_id == 0 or degree_id == 0:
        return []

    if chord.root_pc is None or _normalize_quality(chord.quality) == "N":
        return []

    root = int(chord.root_pc) % 12
    pcs = chord_pitch_classes(root, chord.quality)

    # pick a PC based on degree class
    if degree_id == 1:  # ROOT
        pc = root
    elif degree_id == 2:  # THIRD
        pc = pcs[1] if len(pcs) >= 2 else root
    elif degree_id == 3:  # FIFTH
        pc = pcs[2] if len(pcs) >= 3 else root
    elif degree_id == 4:  # SEVENTH
        pc = pcs[3] if len(pcs) >= 4 else root
    elif degree_id == 5:  # CHORD_TONE
        # prefer a non-root chord tone
        candidates = [p for p in pcs if p != root]
        pc = candidates[0] if candidates else root
    else:  # NONCHORD
        pc = (root + 1) % 12

    pitch = pc_to_pitch(pc, register_id)

    # rhythm_id uses RHY_VOCAB indexing:
    # 0 REST, 1 SUSTAIN, 2 HIT_ON, 3 HIT_OFF, 4 MULTI
    if rhythm_id == 1:  # SUSTAIN
        return [Note(pitch=pitch, start=t0, end=t0 + step_len, velocity=velocity)]
    if rhythm_id == 2:  # HIT_ON
        return [Note(pitch=pitch, start=t0, end=t0 + step_len, velocity=velocity)]
    if rhythm_id == 3:  # HIT_OFF
        return [Note(pitch=pitch, start=t0 + 0.5 * step_len, end=t0 + step_len, velocity=velocity)]
    if rhythm_id == 4:  # MULTI
        return [
            Note(pitch=pitch, start=t0, end=t0 + 0.5 * step_len, velocity=velocity),
            Note(pitch=pitch, start=t0 + 0.5 * step_len, end=t0 + step_len, velocity=velocity),
        ]
    return []


def sample_from_logits(logits: np.ndarray, temperature: float = 0.0, top_k: int = 0) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    x = logits.astype(np.float64) / max(1e-9, float(temperature))
    if top_k and 0 < top_k < x.shape[0]:
        idx = np.argpartition(x, -top_k)[-top_k:]
        mask = np.full_like(x, -1e18)
        mask[idx] = x[idx]
        x = mask

    x = x - np.max(x)
    p = np.exp(x)
    p = p / (np.sum(p) + 1e-12)
    return int(np.random.choice(np.arange(len(p)), p=p))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", type=str, required=True)
    ap.add_argument("--bass_model", type=str, required=True)
    ap.add_argument("--chord_model", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--include_key", action="store_true")
    ap.add_argument("--step_beats", type=float, default=2.0)

    ap.add_argument("--temperature", type=float, default=0.0, help="0=greedy")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--velocity", type=int, default=90)
    args = ap.parse_args()

    pm, info, grid, melody_inst, _sel = load_and_prepare(args.midi)
    melody_notes = extract_melody_notes(melody_inst, grid=None, config=MelodyConfig(quantize_to_beat=False))
    if not melody_notes:
        raise RuntimeError("No melody notes extracted from MIDI.")

    # Chords from ML (step-based if available, else bar-based)
    chords = get_chords_from_ml(
        melody_notes=melody_notes,
        grid=grid,
        duration_seconds=float(info.duration),
        chord_model_path=str(args.chord_model),
        include_key=bool(args.include_key),
        step_beats=float(args.step_beats),
    )

    # Build bass conditioning features (MATCH preprocess_bass.py ORDER)
    step_len = float(args.step_beats) * float(grid.seconds_per_beat)
    step_len = max(1e-6, step_len)
    n_steps = int(np.ceil(max(1e-6, float(info.duration)) / step_len))

    kfeat = key_features(melody_notes) if args.include_key else None
    feats: List[np.ndarray] = []
    for i in range(n_steps):
        t0 = i * step_len
        t1 = t0 + step_len
        mel_feat = melody_step_features(melody_notes, t0, t1, step_len)
        ch = chord_at(chords, t0 + 1e-4)
        ch_feat = chord_step_features(ch)
        pos_feat = step_in_bar_onehot(t0, float(grid.seconds_per_beat), float(args.step_beats))

        if kfeat is not None:
            x = np.concatenate([mel_feat, kfeat, ch_feat, pos_feat], axis=0).astype(np.float32)
        else:
            x = np.concatenate([mel_feat, ch_feat, pos_feat], axis=0).astype(np.float32)

        feats.append(x)

    X = np.stack(feats, axis=0).astype(np.float32)  # (T,F)

    # Load bass model
    model, cfg, dev, meta = load_bass_model(args.bass_model)

    if X.shape[1] != int(cfg.feat_dim):
        raise ValueError(
            f"Feature dim mismatch: model expects feat_dim={cfg.feat_dim}, got {X.shape[1]}.\n"
            f"Most common cause: include_key mismatch.\n"
            f"Try rerun with/without --include_key to match training."
        )

    # Run in chunks of cfg.max_steps (pos embedding length)
    T = X.shape[0]
    max_steps = int(cfg.max_steps)

    bass_notes: List[Note] = []

    start = 0
    while start < T:
        end = min(T, start + max_steps)
        chunk = X[start:end]
        keep = end - start

        x_in = np.zeros((max_steps, X.shape[1]), dtype=np.float32)
        attn = np.zeros((max_steps,), dtype=np.bool_)
        x_in[:keep] = chunk
        attn[:keep] = True

        xb = torch.tensor(x_in[None, :, :], dtype=torch.float32, device=dev)
        attb = torch.tensor(attn[None, :], dtype=torch.bool, device=dev)

        with torch.no_grad():
            deg_logits, reg_logits, rhy_logits = model(xb, attn_mask=attb)
            deg_logits = deg_logits[0, :keep].detach().cpu().numpy()
            reg_logits = reg_logits[0, :keep].detach().cpu().numpy()
            rhy_logits = rhy_logits[0, :keep].detach().cpu().numpy()

        # decode per-step
        for j in range(keep):
            step_idx = start + j
            t0 = step_idx * step_len
            if t0 >= float(info.duration):
                break

            ch = chord_at(chords, t0 + 1e-4)

            if args.temperature > 0:
                deg = sample_from_logits(deg_logits[j], temperature=float(args.temperature), top_k=int(args.top_k))
                rhy = sample_from_logits(rhy_logits[j], temperature=float(args.temperature), top_k=int(args.top_k))
            else:
                deg = int(np.argmax(deg_logits[j]))
                rhy = int(np.argmax(rhy_logits[j]))

            reg = int(np.argmax(reg_logits[j]))

            bass_notes.extend(render_bass_step(
                t0=t0,
                step_len=step_len,
                degree_id=deg,
                register_id=reg,
                rhythm_id=rhy,
                chord=ch,
                velocity=int(args.velocity),
            ))

        start = end

    # Write MIDI with bass track only
    out_pm = pretty_midi.PrettyMIDI(initial_tempo=float(info.tempo_bpm))
    bass_inst = pretty_midi.Instrument(program=33, name="Bass (ML)")  # GM finger bass
    for n in bass_notes:
        bass_inst.notes.append(pretty_midi.Note(
            velocity=int(n.velocity),
            pitch=int(n.pitch),
            start=float(n.start),
            end=float(n.end),
        ))
    out_pm.instruments.append(bass_inst)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_pm.write(str(out_path))
    print(f"Saved: {out_path} | notes={len(bass_notes)}")


if __name__ == "__main__":
    main()
