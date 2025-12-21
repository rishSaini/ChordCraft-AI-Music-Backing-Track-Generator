# app.py
from __future__ import annotations

import inspect
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi
import streamlit as st

from backingtrack.arrange import Arrangement, arrange_backing
from backingtrack.harmony_baseline import ChordEvent, generate_chords
from backingtrack.ml_harmony.steps_infer import ChordSampleConfig, generate_chords_ml_steps
from backingtrack.humanize import HumanizeConfig, humanize_arrangement
from backingtrack.key_detect import estimate_key, key_to_string
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.midi_io import load_and_prepare
from backingtrack.moods import apply_mood_to_key, get_mood, list_moods
from backingtrack.render import RenderConfig, write_midi
from backingtrack.types import BarGrid, Note

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_label(root_pc: int, quality: str, extensions: tuple[int, ...]) -> str:
    name = f"{PC_NAMES[root_pc % 12]}{'' if quality == 'maj' else quality}"
    if 10 in extensions:
        name += "7"
    elif 11 in extensions:
        name += "maj7"
    if 14 in extensions:
        name += "add9"
    return name


# ----------------------------
# Auto-pick helpers (existing)
# ----------------------------
def _median_pitch(inst: pretty_midi.Instrument) -> float:
    pitches = sorted(n.pitch for n in inst.notes)
    if not pitches:
        return 0.0
    m = len(pitches)
    return float(pitches[m // 2]) if (m % 2 == 1) else 0.5 * (pitches[m // 2 - 1] + pitches[m // 2])


def _auto_pick_with_intro(
    pm: pretty_midi.PrettyMIDI,
    info,
    melody_inst: pretty_midi.Instrument,
    sel,
) -> tuple[list[pretty_midi.Instrument], list[int]]:
    """
    Use auto-picked melody_inst as main lead, but also add a short, high-pitch intro lead track if it exists.
    Returns (melody_source_insts, picked_intro_idxs).
    """
    song_end = float(info.duration) if getattr(info, "duration", 0.0) and info.duration > 1e-6 else float(pm.get_end_time())
    main_med = _median_pitch(melody_inst)

    intro_candidates: list[tuple[int, float, int]] = []  # (idx, median_pitch, note_count)

    for idx, inst in enumerate(pm.instruments):
        if idx == sel.instrument_index:
            continue
        if inst.is_drum or not inst.notes:
            continue

        start = min(n.start for n in inst.notes)
        end = max(n.end for n in inst.notes)
        span = max(1e-6, end - start)
        coverage = span / max(1e-6, song_end)

        med = _median_pitch(inst)
        note_count = len(inst.notes)

        if (
            start < 2.0
            and end < 0.25 * song_end
            and coverage < 0.25
            and note_count >= 6
            and med > (main_med + 6)
        ):
            intro_candidates.append((idx, med, note_count))

    intro_candidates.sort(key=lambda x: (-x[1], -x[2]))
    picked_intro_idxs = [idx for (idx, _, _) in intro_candidates[:2]]

    used_indices: list[int] = []
    for i in picked_intro_idxs + [sel.instrument_index]:
        if i not in used_indices:
            used_indices.append(i)

    melody_source_insts = [pm.instruments[i] for i in used_indices]
    return melody_source_insts, picked_intro_idxs


# ----------------------------
# ML Bass (inlined, robust)
# ----------------------------
QUAL_VOCAB_DEFAULT = ["N", "maj", "min", "7", "maj7", "min7", "dim", "sus2", "sus4"]


@dataclass(frozen=True)
class _BassCfg:
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
    cfg = dict(cfg_in)

    # accept alias keys
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

    allowed = {f.name for f in fields(_BassCfg)}
    return {k: v for k, v in cfg.items() if k in allowed}


def _build_causal_mask(T: int, device) -> Any:
    import torch
    return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)


class _BassTransformer:
    # minimal wrapper so we can lazy-import torch only when needed
    def __init__(self, cfg: _BassCfg):
        import torch.nn as nn

        self.cfg = cfg
        self.net = nn.Module()  # placeholder to attach submodules (state_dict compatible)
        self.net.in_proj = nn.Linear(cfg.feat_dim, cfg.d_model)
        self.net.pos = nn.Embedding(cfg.max_steps, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.net.ln = nn.LayerNorm(cfg.d_model)

        self.net.head_degree = nn.Linear(cfg.d_model, cfg.n_degree)
        self.net.head_register = nn.Linear(cfg.d_model, cfg.n_register)
        self.net.head_rhythm = nn.Linear(cfg.d_model, cfg.n_rhythm)

    def to(self, device):
        self.net.to(device)
        return self

    def eval(self):
        self.net.eval()
        return self

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def __call__(self, x, attn_mask=None):
        import torch

        B, T, _ = x.shape
        device = x.device

        h = self.net.in_proj(x)
        idx = torch.arange(T, device=device)
        h = h + self.net.pos(idx)[None, :, :]

        causal = _build_causal_mask(T, device=device)

        pad_mask = None
        if attn_mask is not None:
            pad_mask = ~attn_mask  # transformer expects True where padding

        h = self.net.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.net.ln(h)

        deg = self.net.head_degree(h)
        reg = self.net.head_register(h)
        rhy = self.net.head_rhythm(h)
        return deg, reg, rhy


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _mel_step_feat(melody: Sequence[Note], t0: float, t1: float, step_len: float) -> np.ndarray:
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
    return np.concatenate([hist, np.array([mean_pitch_norm, activity], dtype=np.float32)], axis=0)


def _key_feat(melody: Sequence[Note]) -> np.ndarray:
    k = estimate_key(list(melody))
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0).astype(np.float32)


def _normalize_quality(q: str) -> str:
    q = (q or "").strip().lower()
    if q in ("n", "none", "no_chord", "nochord"):
        return "N"
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
    if q in ("maj", "min", "7", "maj7", "min7", "sus2", "sus4", "dim"):
        return q
    return "maj"


def _chord_at(chords: Sequence[ChordEvent], t: float) -> ChordEvent:
    for c in chords:
        if float(c.start) <= t < float(c.end):
            return c
    return chords[-1]


def _chord_feat(ch: ChordEvent, qual_vocab: list[str]) -> np.ndarray:
    root_oh = np.zeros(12, dtype=np.float32)
    root_oh[int(ch.root_pc) % 12] = 1.0

    q = _normalize_quality(str(ch.quality))
    qual_to_i = {qq: i for i, qq in enumerate(qual_vocab)}
    qual_oh = np.zeros(len(qual_vocab), dtype=np.float32)
    qual_oh[qual_to_i.get(q, qual_to_i.get("N", 0))] = 1.0
    return np.concatenate([root_oh, qual_oh], axis=0).astype(np.float32)


def _pos_feat(t0: float, spb: float, pos_bins: int) -> np.ndarray:
    beats = (t0 / max(1e-9, spb)) % 4.0
    idx = int(np.floor(beats / (4.0 / float(pos_bins)))) % pos_bins
    oh = np.zeros(pos_bins, dtype=np.float32)
    oh[idx] = 1.0
    return oh


def _sample_id(logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
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
    return int(rng.choice(np.arange(len(p)), p=p))


def _chord_pcs(root_pc: int, quality: str, extensions: Tuple[int, ...]) -> Tuple[int, ...]:
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
    pcs += [(r + int(iv)) % 12 for iv in extensions]
    out: List[int] = []
    for pc in pcs:
        if pc not in out:
            out.append(int(pc))
    return tuple(out)


def _pc_to_pitch(pc: int, register_id: int, n_register: int) -> int:
    if n_register <= 1:
        center = 50
    else:
        centers = [40, 50, 60]
        center = centers[int(np.clip(register_id, 0, len(centers) - 1))]
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


def _render_step(
    t0: float,
    step_len: float,
    degree_id: int,
    register_id: int,
    rhythm_id: int,
    chord: ChordEvent,
    velocity: int,
    n_degree: int,
    n_rhythm: int,
) -> List[Note]:
    # rhythm handling:
    # - if n_rhythm==2: 0 rest, 1 hit
    # - if n_rhythm>=5: 0 REST, 1 SUSTAIN, 2 HIT_ON, 3 HIT_OFF, 4 MULTI
    if n_rhythm == 2:
        if rhythm_id == 0:
            return []
        rhythm_mode = "HIT_ON"
    else:
        if rhythm_id == 0:
            return []
        rhythm_mode = {1: "SUSTAIN", 2: "HIT_ON", 3: "HIT_OFF", 4: "MULTI"}.get(rhythm_id, "HIT_ON")

    pcs = _chord_pcs(int(chord.root_pc), str(chord.quality), tuple(getattr(chord, "extensions", ()) or ()))
    if not pcs:
        return []

    root = int(chord.root_pc) % 12

    # degree handling (assumes 0=REST if present)
    if n_degree >= 7:
        # 0 REST, 1 ROOT, 2 THIRD, 3 FIFTH, 4 SEVENTH, 5 CHORD_TONE, 6 NONCHORD
        if degree_id == 0:
            return []
        if degree_id == 1:
            pc = root
        elif degree_id == 2:
            pc = pcs[1] if len(pcs) >= 2 else root
        elif degree_id == 3:
            pc = pcs[2] if len(pcs) >= 3 else root
        elif degree_id == 4:
            pc = pcs[3] if len(pcs) >= 4 else root
        elif degree_id == 5:
            cand = [p for p in pcs if p != root]
            pc = cand[0] if cand else root
        else:
            pc = (root + 1) % 12
    else:
        # fallback: treat any nonzero degree as root-ish
        if degree_id == 0:
            return []
        pc = root

    pitch = _pc_to_pitch(pc, int(register_id), n_register=3)

    if rhythm_mode in ("SUSTAIN", "HIT_ON"):
        return [Note(pitch=pitch, start=t0, end=t0 + step_len * 0.92, velocity=velocity)]
    if rhythm_mode == "HIT_OFF":
        return [Note(pitch=pitch, start=t0 + 0.5 * step_len, end=t0 + step_len * 0.98, velocity=velocity)]
    if rhythm_mode == "MULTI":
        return [
            Note(pitch=pitch, start=t0, end=t0 + 0.5 * step_len, velocity=velocity),
            Note(pitch=pitch, start=t0 + 0.5 * step_len, end=t0 + step_len * 0.98, velocity=velocity),
        ]
    return []


@st.cache_resource(show_spinner=False)
def _load_bass_model_cached(model_path: str):
    import torch

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Bass model not found: {p}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    cfg_dict = _normalize_bass_cfg(dict(ckpt["cfg"]))
    cfg = _BassCfg(**cfg_dict)

    model = _BassTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    meta = ckpt.get("meta", {}) or {}
    return model, cfg, dev, meta


def generate_bass_ml(
    *,
    model_path: str,
    melody_notes: List[Note],
    chords: List[ChordEvent],
    grid: BarGrid,
    duration_seconds: float,
    include_key: bool,
    step_beats: float,
    temperature: float,
    top_k: int,
    seed: Optional[int],
    velocity: int,
) -> List[Note]:
    model, cfg, dev, meta = _load_bass_model_cached(model_path)

    qual_vocab = list(meta.get("qual_vocab") or QUAL_VOCAB_DEFAULT)

    spb = float(grid.seconds_per_beat)
    step_len = max(1e-6, float(step_beats) * spb)
    n_steps = int(np.ceil(max(1e-6, float(duration_seconds)) / step_len))

    # Choose pos_bins to match feat_dim (try a few common options)
    mel_dim = 14
    key_dim = 14 if include_key else 0
    chord_dim = 12 + len(qual_vocab)

    pos_bins_candidates = [2, 4, 1, 8]
    pos_bins = None
    for b in pos_bins_candidates:
        if mel_dim + key_dim + chord_dim + b == int(cfg.feat_dim):
            pos_bins = b
            break
    if pos_bins is None:
        raise ValueError(
            f"Bass feat_dim mismatch. Model expects feat_dim={cfg.feat_dim}, "
            f"but schema gives {mel_dim}+{key_dim}+{chord_dim}+pos_bins.\n"
            f"Try toggling include_key or retrain bass with the same features. "
            f"(qual_vocab_len={len(qual_vocab)})"
        )

    kfeat = _key_feat(melody_notes) if include_key else None

    X = np.zeros((n_steps, int(cfg.feat_dim)), dtype=np.float32)
    for i in range(n_steps):
        t0 = float(i) * step_len
        t1 = t0 + step_len
        mfeat = _mel_step_feat(melody_notes, t0, t1, step_len)
        ch = _chord_at(chords, t0 + 1e-4)
        cfeat = _chord_feat(ch, qual_vocab=qual_vocab)
        pfeat = _pos_feat(t0, spb, pos_bins=pos_bins)

        if kfeat is not None:
            x = np.concatenate([mfeat, kfeat, cfeat, pfeat], axis=0).astype(np.float32)
        else:
            x = np.concatenate([mfeat, cfeat, pfeat], axis=0).astype(np.float32)
        X[i] = x

    import torch

    rng = np.random.default_rng(seed if seed is not None else None)

    T = X.shape[0]
    max_steps = int(cfg.max_steps)

    out_notes: List[Note] = []

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

        for j in range(keep):
            step_idx = start + j
            t0 = float(step_idx) * step_len
            if t0 >= float(duration_seconds):
                break

            ch = _chord_at(chords, t0 + 1e-4)

            deg_id = _sample_id(deg_logits[j], temperature=float(temperature), top_k=int(top_k), rng=rng)
            reg_id = int(np.argmax(reg_logits[j]))
            rhy_id = _sample_id(rhy_logits[j], temperature=float(temperature), top_k=int(top_k), rng=rng)

            out_notes.extend(
                _render_step(
                    t0=t0,
                    step_len=step_len,
                    degree_id=int(deg_id),
                    register_id=int(reg_id),
                    rhythm_id=int(rhy_id),
                    chord=ch,
                    velocity=int(velocity),
                    n_degree=int(cfg.n_degree),
                    n_rhythm=int(cfg.n_rhythm),
                )
            )

        start = end

    # safe-sort
    out_notes.sort(key=lambda n: (n.start, n.pitch))
    return out_notes


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Backing Track Maker", page_icon="🎼", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1200px; }
      .stButton>button { border-radius: 14px; padding: 0.75rem 1.1rem; font-weight: 650; }
      .card { border: 1px solid rgba(255,255,255,0.10); border-radius: 18px; padding: 16px 18px; background: rgba(255,255,255,0.04); }
      .muted { opacity: 0.75; }
      code { font-size: 0.95em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎼 AI Backing Track Maker")
st.caption("Upload a MIDI melody → generate a backing track (bass/pad/drums) → download a new multi-track MIDI.")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Upload MIDI")
    uploaded = st.file_uploader("MIDI file (.mid / .midi)", type=["mid", "midi"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
    st.subheader("2) Controls")

    moods = list_moods()
    mood_name = st.selectbox("Mood", moods, index=moods.index("neutral") if "neutral" in moods else 0)

    st.markdown("**Harmony (chords)**")
    harmony_mode = st.selectbox("Chord generator", ["baseline (rules)", "ml (transformer)"], index=0)

    # chord defaults
    chord_model_path = "data/ml/chord_model_new.pt"
    chord_step_beats = 2.0
    chord_include_key = True
    chord_stochastic = False
    chord_temp = 1.0
    chord_top_k = 12
    chord_repeat_penalty = 1.2
    chord_change_penalty = 0.15

    if harmony_mode.startswith("ml"):
        chord_model_path = st.text_input("Chord model path", value=chord_model_path)
        chord_step_beats = float(st.selectbox("Chord step size (beats)", [1.0, 2.0, 4.0], index=1))
        chord_include_key = st.toggle("Include key features (recommended)", value=True)

        chord_stochastic = st.toggle("Stochastic chords (more variety)", value=False)
        chord_temp = st.slider("Chord temperature", 0.7, 1.6, 1.0, 0.01)
        chord_top_k = st.slider("Chord top-k (0 = no top-k)", 0, 40, 12, 1)
        chord_repeat_penalty = st.slider("Chord repeat penalty", 0.0, 3.0, 1.2, 0.05)
        chord_change_penalty = st.slider(
            "Chord smoothness (change penalty)",
            0.0,
            0.6,
            0.15,
            0.01,
            disabled=chord_stochastic,
        )
        bars_per_chord = st.slider("Bars per chord (baseline only)", 1, 4, 1, 1, disabled=True)
    else:
        bars_per_chord = st.slider("Bars per chord", 1, 4, 1, 1)

    quantize_melody = st.toggle("Quantize melody to beat grid", value=False)

    st.divider()
    st.markdown("**Humanize**")
    humanize = st.toggle("Humanize timing/velocity", value=True)
    jitter_ms = st.slider("Timing jitter (ms)", 0.0, 35.0, 15.0, 1.0)
    vel_jitter = st.slider("Velocity jitter", 0, 20, 8, 1)
    swing = st.slider("Swing (0..1)", 0.0, 0.6, 0.15, 0.01)

    seed_value = st.number_input("Seed (optional)", value=0, step=1)
    use_seed = st.toggle("Use seed", value=False)
    seed: Optional[int] = int(seed_value) if use_seed else None

    structure_mode = "auto" if st.toggle("Auto song sections (intro/verse/chorus/outro)", value=True) else "none"

    st.divider()
    st.markdown("**Backing tracks**")
    make_bass = st.toggle("Bass", value=True)
    make_pad = st.toggle("Pad", value=True)
    make_drums = st.toggle("Drums", value=True)

    # NEW: bass mode + controls
    bass_mode = st.selectbox("Bass generator", ["rules", "ml"], index=1 if make_bass else 0, disabled=not make_bass)
    bass_model_path = "data/ml/bass_model.pt"
    bass_step_beats = 2.0
    bass_include_key = True
    bass_temp = 0.0
    bass_top_k = 0
    bass_velocity = 90

    if make_bass and bass_mode == "ml":
        bass_model_path = st.text_input("Bass model path", value=bass_model_path)
        bass_step_beats = float(st.selectbox("Bass step size (beats)", [1.0, 2.0, 4.0], index=1))
        bass_include_key = st.toggle("Bass include key features", value=True)
        bass_temp = st.slider("Bass temperature (0 = greedy)", 0.0, 1.6, 0.0, 0.01)
        bass_top_k = st.slider("Bass top-k (0 = off)", 0, 40, 0, 1)
        bass_velocity = st.slider("Bass velocity", 40, 120, 90, 1)

    drums_mode = st.selectbox("Drums", ["rules", "ml"], index=0)
    ml_temp = st.slider("ML drum temperature", 0.8, 1.4, 1.05, 0.01)

    st.markdown("</div>", unsafe_allow_html=True)

    generate_btn = st.button("✨ Generate backing track", use_container_width=True, disabled=(uploaded is None))

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output")
    st.markdown(
        '<p class="muted">Once generated, you’ll see detected key, chosen melody track(s), chord progression preview, and a download button.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def run_pipeline(
    midi_bytes: bytes,
    *,
    mood_name: str,
    harmony_mode: str,
    chord_model_path: str,
    chord_step_beats: float,
    chord_include_key: bool,
    chord_stochastic: bool,
    chord_temp: float,
    chord_top_k: int,
    chord_repeat_penalty: float,
    chord_change_penalty: float,
    bars_per_chord: int,
    quantize_melody: bool,
    make_bass: bool,
    make_pad: bool,
    make_drums: bool,
    bass_mode: str,
    bass_model_path: str,
    bass_step_beats: float,
    bass_include_key: bool,
    bass_temp: float,
    bass_top_k: int,
    bass_velocity: int,
    melody_track_indices: Optional[list[int]],
    seed: Optional[int],
    structure_mode: str,
    drums_mode: str,
    ml_temp: float,
    humanize: bool,
    jitter_ms: float,
    vel_jitter: int,
    swing: float,
) -> tuple[Path, dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_bytes)
        in_path = Path(f.name)

    first_idx = melody_track_indices[0] if melody_track_indices else None
    pm, info, grid, melody_inst, sel = load_and_prepare(in_path, melody_instrument_index=first_idx)

    picked_intro_idxs: list[int] = []
    used_melody_indices: list[int] = []

    # Decide which instruments are the "lead"
    if melody_track_indices:
        valid: list[int] = []
        for i in melody_track_indices:
            if 0 <= i < len(pm.instruments):
                valid.append(i)
        used_melody_indices = sorted(set(valid)) if valid else [sel.instrument_index]
        melody_source_insts = [pm.instruments[i] for i in used_melody_indices]
    else:
        melody_source_insts, picked_intro_idxs = _auto_pick_with_intro(pm, info, melody_inst, sel)
        for i, inst in enumerate(pm.instruments):
            if any(inst is x for x in melody_source_insts):
                used_melody_indices.append(i)

    # Combine selected lead instruments for analysis
    analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
    analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
    analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

    mel_cfg = MelodyConfig(quantize_to_beat=quantize_melody)
    melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=mel_cfg)
    if not melody_notes:
        raise RuntimeError("No melody notes extracted. Try selecting a different melody track (or multiple tracks).")

    mood = get_mood(mood_name)
    raw_key = estimate_key(melody_notes)
    key = apply_mood_to_key(raw_key, mood)

    # Chords
    if str(harmony_mode).startswith("ml"):
        chords = generate_chords_ml_steps(
            melody_notes=melody_notes,
            grid=grid,
            duration_seconds=float(info.duration),
            model_path=str(chord_model_path),
            cfg=ChordSampleConfig(
                step_beats=float(chord_step_beats),
                include_key=bool(chord_include_key),
                stochastic=bool(chord_stochastic),
                temperature=float(chord_temp),
                top_k=int(chord_top_k),
                repeat_penalty=float(chord_repeat_penalty),
                change_penalty=float(chord_change_penalty),
                seed=seed,
            ),
        )
    else:
        chords = generate_chords(
            key=key,
            grid=grid,
            duration_seconds=info.duration,
            mood=mood,
            melody_notes=melody_notes,
            bars_per_chord=bars_per_chord,
        )

    # Arrange pad/drums (and rules bass unless ML bass selected)
    arrangement = arrange_backing(
        chords=chords,
        grid=grid,
        mood=mood,
        make_bass=bool(make_bass and bass_mode != "ml"),
        make_pad=make_pad,
        make_drums=make_drums,
        seed=seed,
        structure_mode=structure_mode,
        drums_mode=drums_mode,
        ml_drums_model_path="data/ml/drum_model.pt",
        ml_drums_temperature=ml_temp,
    )

    # ML bass override (before humanize)
    if make_bass and bass_mode == "ml":
        bass_notes = generate_bass_ml(
            model_path=str(bass_model_path),
            melody_notes=melody_notes,
            chords=list(chords),
            grid=grid,
            duration_seconds=float(info.duration),
            include_key=bool(bass_include_key),
            step_beats=float(bass_step_beats),
            temperature=float(bass_temp),
            top_k=int(bass_top_k),
            seed=seed,
            velocity=int(bass_velocity),
        )
        tracks = dict(arrangement.tracks)
        tracks["bass"] = bass_notes
        arrangement = Arrangement(tracks=tracks)

    if humanize:
        arrangement = humanize_arrangement(
            arrangement,
            grid,
            HumanizeConfig(timing_jitter_ms=jitter_ms, velocity_jitter=vel_jitter, swing=swing, seed=seed),
        )

    out_path = Path(tempfile.mkstemp(suffix=".mid")[1])
    render_cfg = RenderConfig(melody_program=int(melody_source_insts[0].program))

    # Render lead from original instruments + backing tracks
    write_midi(
        out_path,
        [],
        arrangement,
        info,
        config=render_cfg,
        melody_source_insts=melody_source_insts,
    )

    meta = {
        "info": info,
        "selection": sel,
        "selected_melody_indices": melody_track_indices,
        "used_melody_indices": used_melody_indices,
        "auto_intro_indices": picked_intro_idxs,
        "used_melody_track_names": [inst.name or "(unnamed)" for inst in melody_source_insts],
        "melody_note_count": len(melody_notes),
        "raw_key": raw_key,
        "key": key,
        "mood": mood,
        "harmony_mode": harmony_mode,
        "chord_model_path": str(chord_model_path),
        "chord_step_beats": float(chord_step_beats),
        "chord_stochastic": bool(chord_stochastic),
        "chord_temperature": float(chord_temp),
        "chord_top_k": int(chord_top_k),
        "chord_repeat_penalty": float(chord_repeat_penalty),
        "chord_change_penalty": float(chord_change_penalty),
        "bass_mode": bass_mode,
        "bass_model_path": str(bass_model_path),
        "bass_step_beats": float(bass_step_beats),
        "bass_include_key": bool(bass_include_key),
        "bass_temperature": float(bass_temp),
        "bass_top_k": int(bass_top_k),
        "chords": chords,
        "arrangement_counts": {k: len(v) for k, v in arrangement.tracks.items()},
        "instrument_list": [
            {"idx": i, "name": (inst.name or f"Instrument {i}"), "is_drum": inst.is_drum, "notes": len(inst.notes)}
            for i, inst in enumerate(pm.instruments)
        ],
    }
    return out_path, meta


# --- Melody track picker (multi-select) ---
melody_track_indices: Optional[list[int]] = None

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(uploaded.getvalue())
        tmp_path = Path(f.name)

    pm_preview, info_preview, grid_preview, melody_inst_preview, sel_preview = load_and_prepare(tmp_path, melody_instrument_index=None)
    _, intro_preview = _auto_pick_with_intro(pm_preview, info_preview, melody_inst_preview, sel_preview)

    with left:
        st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
        st.subheader("3) Melody track(s)")

        use_auto = st.toggle("Auto-pick melody track", value=True)

        options: list[str] = []
        default_label: Optional[str] = None
        for i, inst in enumerate(pm_preview.instruments):
            nm = inst.name or f"Instrument {i}"
            tag = "DRUMS" if inst.is_drum else "INST"
            label = f"{i}: {nm}  ·  {tag}  ·  notes={len(inst.notes)}"
            options.append(label)
            if i == sel_preview.instrument_index:
                default_label = label

        if use_auto:
            melody_track_indices = None
            if intro_preview:
                st.caption(
                    f"Auto-picked main: idx={sel_preview.instrument_index}, name='{sel_preview.instrument_name}' "
                    f"(+ intro tracks: {intro_preview})"
                )
            else:
                st.caption(f"Auto-picked: idx={sel_preview.instrument_index}, name='{sel_preview.instrument_name}'")
        else:
            picked = st.multiselect(
                "Choose melody instrument(s) (pick ALL tracks that contain the lead)",
                options=options,
                default=[default_label] if default_label else [],
            )
            melody_track_indices = [int(x.split(":")[0].strip()) for x in picked] if picked else None

        with st.expander("Show instrument list"):
            st.json(
                [
                    {
                        "idx": i,
                        "name": (inst.name or f"Instrument {i}"),
                        "is_drum": inst.is_drum,
                        "notes": len(inst.notes),
                    }
                    for i, inst in enumerate(pm_preview.instruments)
                ]
            )

        st.markdown("</div>", unsafe_allow_html=True)


if generate_btn and uploaded is not None:
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Output")

        try:
            with st.spinner("Generating backing track..."):
                out_path, meta = run_pipeline(
                    midi_bytes=uploaded.getvalue(),
                    mood_name=mood_name,
                    harmony_mode=harmony_mode,
                    chord_model_path=chord_model_path,
                    chord_step_beats=float(chord_step_beats),
                    chord_include_key=bool(chord_include_key),
                    chord_stochastic=bool(chord_stochastic),
                    chord_temp=float(chord_temp),
                    chord_top_k=int(chord_top_k),
                    chord_repeat_penalty=float(chord_repeat_penalty),
                    chord_change_penalty=float(chord_change_penalty),
                    bars_per_chord=int(bars_per_chord),
                    quantize_melody=quantize_melody,
                    make_bass=make_bass,
                    make_pad=make_pad,
                    make_drums=make_drums,
                    bass_mode=bass_mode,
                    bass_model_path=bass_model_path,
                    bass_step_beats=float(bass_step_beats),
                    bass_include_key=bool(bass_include_key),
                    bass_temp=float(bass_temp),
                    bass_top_k=int(bass_top_k),
                    bass_velocity=int(bass_velocity),
                    melody_track_indices=melody_track_indices,
                    seed=seed,
                    structure_mode=structure_mode,
                    drums_mode=drums_mode,
                    ml_temp=ml_temp,
                    humanize=humanize,
                    jitter_ms=jitter_ms,
                    vel_jitter=int(vel_jitter),
                    swing=float(swing),
                )
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            raise

        info = meta["info"]
        sel = meta["selection"]

        st.success("Done ✅")

        colA, colB, colC = st.columns(3)
        colA.metric("Tempo (BPM)", f"{info.tempo_bpm:.1f}")
        colB.metric("Time Signature", f"{info.time_signature.numerator}/{info.time_signature.denominator}")
        colC.metric("Duration (s)", f"{info.duration:.1f}")

        if meta["selected_melody_indices"]:
            st.markdown(f"**Melody tracks (manual):** {meta['selected_melody_indices']}")
        else:
            if meta["auto_intro_indices"]:
                st.markdown(
                    f"**Melody tracks (auto):** main idx={sel.instrument_index} · `{sel.instrument_name}` "
                    f"(+ intro: {meta['auto_intro_indices']})"
                )
            else:
                st.markdown(f"**Melody track (auto):** idx={sel.instrument_index} · `{sel.instrument_name}`")

        st.markdown(f"**Used melody indices:** {meta['used_melody_indices']}")
        st.markdown(f"**Melody notes extracted (for analysis):** `{meta['melody_note_count']}`")

        st.markdown(f"**Detected key:** {key_to_string(meta['raw_key'])}")
        if meta["key"] != meta["raw_key"]:
            st.markdown(f"**After mood '{meta['mood'].name}' bias:** {key_to_string(meta['key'])}")

        st.markdown("**Backing note counts:**")
        st.json(meta["arrangement_counts"])

        chords = meta["chords"]
        preview = " · ".join(chord_label(c.root_pc, c.quality, c.extensions) for c in chords[:8])
        st.markdown("**Chord progression preview:**")
        st.code(preview if preview else "(none)")

        st.markdown(f"**Bass mode:** `{meta['bass_mode']}`")
        if meta["bass_mode"] == "ml":
            st.markdown(f"**Bass model:** `{meta['bass_model_path']}`")

        midi_out_bytes = out_path.read_bytes()
        st.download_button(
            label="⬇️ Download generated MIDI",
            data=midi_out_bytes,
            file_name="backing_track.mid",
            mime="audio/midi",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
