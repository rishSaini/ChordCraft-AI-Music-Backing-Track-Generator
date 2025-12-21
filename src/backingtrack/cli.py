# src/backingtrack/cli.py
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi
import typer

from .midi_io import load_and_prepare
from .melody import MelodyConfig, extract_melody_notes
from .key_detect import estimate_key, key_to_string
from .moods import apply_mood_to_key, get_mood, list_moods
from .harmony_baseline import ChordEvent, generate_chords
from .ml_harmony.steps_infer import ChordSampleConfig, generate_chords_ml_steps
from .arrange import Arrangement, arrange_backing
from .render import RenderConfig, write_midi
from .humanize import HumanizeConfig, humanize_arrangement


app = typer.Typer(add_completion=False, help="AI MIDI backing-track generator (baseline + ML modules).")


def _parse_indices(csv: Optional[str]) -> list[int]:
    if not csv:
        return []
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _median_pitch(inst: pretty_midi.Instrument) -> float:
    pitches = sorted(n.pitch for n in inst.notes)
    if not pitches:
        return 0.0
    m = len(pitches)
    return float(pitches[m // 2]) if (m % 2 == 1) else 0.5 * (pitches[m // 2 - 1] + pitches[m // 2])


# ----------------------------
# ML Bass (same logic as app.py)
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


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _mel_step_feat(melody: Sequence["Note"], t0: float, t1: float, step_len: float) -> np.ndarray:
    from .types import Note  # local import
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


def _key_feat(melody: Sequence["Note"]) -> np.ndarray:
    from .key_detect import estimate_key
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
) -> List["Note"]:
    from .types import Note

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

    if n_degree >= 7:
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


def generate_bass_ml(
    *,
    model_path: str,
    melody_notes: List["Note"],
    chords: List[ChordEvent],
    grid: "BarGrid",
    duration_seconds: float,
    include_key: bool,
    step_beats: float,
    temperature: float,
    top_k: int,
    seed: Optional[int],
    velocity: int,
) -> List["Note"]:
    import torch

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Bass model not found: {p}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    cfg_dict = _normalize_bass_cfg(dict(ckpt["cfg"]))
    cfg = _BassCfg(**cfg_dict)

    # build model matching training (pos emb + causal transformer + 3 heads)
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self, cfg: _BassCfg):
            super().__init__()
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

        def forward(self, x, attn_mask=None):
            B, T, _ = x.shape
            device = x.device
            h = self.in_proj(x)
            idx = torch.arange(T, device=device)
            h = h + self.pos(idx)[None, :, :]
            causal = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
            pad_mask = None
            if attn_mask is not None:
                pad_mask = ~attn_mask
            h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
            h = self.ln(h)
            return self.head_degree(h), self.head_register(h), self.head_rhythm(h)

    model = _M(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    meta = ckpt.get("meta", {}) or {}
    qual_vocab = list(meta.get("qual_vocab") or QUAL_VOCAB_DEFAULT)

    spb = float(grid.seconds_per_beat)
    step_len = max(1e-6, float(step_beats) * spb)
    n_steps = int(np.ceil(max(1e-6, float(duration_seconds)) / step_len))

    mel_dim = 14
    key_dim = 14 if include_key else 0
    chord_dim = 12 + len(qual_vocab)

    pos_bins = None
    for b in [2, 4, 1, 8]:
        if mel_dim + key_dim + chord_dim + b == int(cfg.feat_dim):
            pos_bins = b
            break
    if pos_bins is None:
        raise ValueError(f"feat_dim mismatch: model expects {cfg.feat_dim}, cannot match schema.")

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
            X[i] = np.concatenate([mfeat, kfeat, cfeat, pfeat], axis=0).astype(np.float32)
        else:
            X[i] = np.concatenate([mfeat, cfeat, pfeat], axis=0).astype(np.float32)

    rng = np.random.default_rng(seed if seed is not None else None)

    T = X.shape[0]
    max_steps = int(cfg.max_steps)
    out_notes: List["Note"] = []

    from .types import Note, BarGrid

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

    out_notes.sort(key=lambda n: (n.start, n.pitch))
    return out_notes


# ----------------------------
# CLI command
# ----------------------------
@app.command()
def generate(
    input_midi: Path = typer.Argument(..., exists=True, readable=True, help="Input MIDI file (.mid/.midi)"),
    output_midi: Path = typer.Option(Path("data/generated/out.mid"), "--out", "-o", help="Output MIDI path"),
    mood: str = typer.Option("neutral", "--mood", "-m", help=f"Mood preset. Options: {', '.join(list_moods())}"),
    melody_tracks: Optional[str] = typer.Option(
        None,
        "--melody-tracks",
        help="Comma-separated instrument indices to render as the lead (e.g. 0,2,5). If omitted, auto-pick is used.",
    ),
    bars_per_chord: int = typer.Option(1, "--bars-per-chord", help="How many bars each chord lasts"),
    harmony_mode: str = typer.Option("baseline", "--harmony-mode", help="Chords: baseline | ml"),
    chord_model: Path = typer.Option(Path("data/ml/chord_model_new.pt"), "--chord-model", help="Path to ML chord model (.pt)"),
    chord_step_beats: float = typer.Option(2.0, "--chord-step-beats", help="Chord model step size in beats"),
    chord_include_key: bool = typer.Option(True, "--chord-include-key/--no-chord-include-key", help="Include key features in ML chord model"),
    chord_stochastic: bool = typer.Option(False, "--chord-stochastic/--no-chord-stochastic", help="Sample chords stochastically"),
    chord_temp: float = typer.Option(1.0, "--chord-temp", help="Chord model temperature (if harmony_mode=ml)"),
    chord_top_k: int = typer.Option(12, "--chord-top-k", help="Chord model top-k (0 = no top-k)"),
    chord_repeat_penalty: float = typer.Option(1.2, "--chord-repeat-penalty", help="Penalty for repeating the same chord (if stochastic)"),
    chord_change_penalty: float = typer.Option(0.15, "--chord-change-penalty", help="Change penalty for smoothing (if not stochastic)"),
    # NEW: bass ML options
    bass_mode: str = typer.Option("rules", "--bass-mode", help="Bass: rules | ml"),
    bass_model: Path = typer.Option(Path("data/ml/bass_model.pt"), "--bass-model", help="Path to ML bass model (.pt)"),
    bass_step_beats: float = typer.Option(2.0, "--bass-step-beats", help="Bass step size in beats (must match training)"),
    bass_include_key: bool = typer.Option(True, "--bass-include-key/--no-bass-include-key", help="Include key features for bass"),
    bass_temp: float = typer.Option(0.0, "--bass-temp", help="Bass temperature (0 = greedy)"),
    bass_top_k: int = typer.Option(0, "--bass-top-k", help="Bass top-k (0 = off)"),
    bass_velocity: int = typer.Option(90, "--bass-velocity", help="Bass velocity (used for generated notes)"),
    quantize_melody: bool = typer.Option(False, "--quantize-melody", help="Quantize melody note times to the beat grid"),
    no_drums: bool = typer.Option(False, "--no-drums", help="Disable drum track generation"),
    no_bass: bool = typer.Option(False, "--no-bass", help="Disable bass track generation"),
    no_pad: bool = typer.Option(False, "--no-pad", help="Disable pad track generation"),
    humanize: bool = typer.Option(True, "--humanize/--no-humanize", help="Apply humanization"),
    jitter_ms: float = typer.Option(15.0, "--jitter-ms", help="Timing jitter in milliseconds"),
    vel_jitter: int = typer.Option(8, "--vel-jitter", help="Velocity jitter"),
    swing: float = typer.Option(0.15, "--swing", help="Swing amount 0..1"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    structure: str = typer.Option("none", "--structure", help="Song structure: none | auto"),
    drums_mode: str = typer.Option("rules", "--drums-mode", help="Drums: rules | ml"),
    ml_temp: float = typer.Option(1.05, "--ml-temp", help="ML drum temperature (if drums_mode=ml)"),
):
    requested = _parse_indices(melody_tracks)
    first_idx = requested[0] if requested else None

    pm, info, grid, melody_inst, sel = load_and_prepare(input_midi, melody_instrument_index=first_idx)

    # Decide which instruments are the "lead"
    if requested:
        for i in requested:
            if i < 0 or i >= len(pm.instruments):
                raise typer.BadParameter(f"melody track index {i} out of range (0..{len(pm.instruments)-1})")
        melody_source_insts = [pm.instruments[i] for i in requested]
        typer.echo(f"Using melody tracks (lead): {requested}")
    else:
        melody_source_insts = [melody_inst]
        song_end = float(info.duration) if info.duration > 1e-6 else float(pm.get_end_time())
        main_med = _median_pitch(melody_inst)

        intro_candidates: list[tuple[int, float, int]] = []
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
            if start < 2.0 and end < 0.25 * song_end and coverage < 0.25 and note_count >= 6 and med > (main_med + 6):
                intro_candidates.append((idx, med, note_count))

        intro_candidates.sort(key=lambda x: (-x[1], -x[2]))
        picked_intro_idxs = [idx for (idx, _, _) in intro_candidates[:2]]
        if picked_intro_idxs:
            melody_source_insts = [pm.instruments[i] for i in picked_intro_idxs] + melody_source_insts

        typer.echo(
            f"Auto-picked melody track: idx={sel.instrument_index}, name='{sel.instrument_name}', is_drum={sel.is_drum}"
            + (f" | plus intro tracks: {picked_intro_idxs}" if picked_intro_idxs else "")
        )

    typer.echo(f"Tempo: {info.tempo_bpm:.2f} BPM | Time signature: {info.time_signature.numerator}/{info.time_signature.denominator}")
    typer.echo(f"Duration: {info.duration:.2f}s")

    # Analysis instrument over all selected lead tracks
    analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
    analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
    analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

    mel_cfg = MelodyConfig(quantize_to_beat=quantize_melody)
    melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=mel_cfg)
    if not melody_notes:
        raise typer.Exit(code=1)

    typer.echo(f"Melody notes extracted: {len(melody_notes)}")

    mood_preset = get_mood(mood)
    raw_key = estimate_key(melody_notes)
    key = apply_mood_to_key(raw_key, mood_preset)

    typer.echo(f"Detected key: {key_to_string(raw_key)}")
    if key != raw_key:
        typer.echo(f"After mood '{mood_preset.name}' bias: {key_to_string(key)}")

    # Chords
    if str(harmony_mode).startswith("ml"):
        chords = generate_chords_ml_steps(
            melody_notes=melody_notes,
            grid=grid,
            duration_seconds=float(info.duration),
            model_path=str(chord_model),
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
        typer.echo(f"Chords generated: {len(chords)} (ML)")
    else:
        chords = generate_chords(
            key=key,
            grid=grid,
            duration_seconds=info.duration,
            mood=mood_preset,
            melody_notes=melody_notes,
            bars_per_chord=bars_per_chord,
        )
        typer.echo(f"Chords generated: {len(chords)} (baseline)")

    # Arrange pad/drums (rules bass unless ML bass)
    arrangement = arrange_backing(
        chords=chords,
        grid=grid,
        mood=mood_preset,
        make_bass=(not no_bass) and (bass_mode != "ml"),
        make_pad=not no_pad,
        make_drums=not no_drums,
        seed=seed,
        structure_mode=structure,
        drums_mode=drums_mode,
        ml_drums_model_path="data/ml/drum_model.pt",
        ml_drums_temperature=ml_temp,
    )

    # ML bass override (before humanize)
    if (not no_bass) and bass_mode == "ml":
        from .types import Note, BarGrid  # for type resolution
        bass_notes = generate_bass_ml(
            model_path=str(bass_model),
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
        typer.echo(f"ML bass notes: {len(bass_notes)}")

    if humanize:
        arrangement = humanize_arrangement(
            arrangement,
            grid,
            HumanizeConfig(timing_jitter_ms=jitter_ms, velocity_jitter=vel_jitter, swing=swing, seed=seed),
        )

    counts = {k: len(v) for k, v in arrangement.tracks.items()}
    typer.echo(f"Backing note counts: {counts}")

    render_cfg = RenderConfig(melody_program=int(melody_source_insts[0].program))
    write_midi(output_midi, [], arrangement, info, config=render_cfg, melody_source_insts=melody_source_insts)
    typer.echo(f"Wrote: {output_midi.resolve()}")


def main():
    app()


if __name__ == "__main__":
    main()
