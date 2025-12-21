# app.py
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pretty_midi
import streamlit as st

from backingtrack.arrange import arrange_backing
from backingtrack.harmony_baseline import generate_chords
from backingtrack.ml_harmony.steps_infer import ChordSampleConfig, generate_chords_ml_steps
from backingtrack.humanize import HumanizeConfig, humanize_arrangement
from backingtrack.key_detect import estimate_key, key_to_string
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.midi_io import load_and_prepare
from backingtrack.moods import apply_mood_to_key, get_mood, list_moods
from backingtrack.render import RenderConfig, write_midi

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
# Auto settings (new)
# ----------------------------
@st.cache_data(show_spinner=False)
def recommend_settings(midi_bytes: bytes) -> Dict[str, Any]:
    """
    Heuristic 'auto' settings. No training required.
    Returns a dict of session_state keys -> values.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_bytes)
        tmp_path = Path(f.name)

    pm, info, grid, melody_inst, sel = load_and_prepare(tmp_path, melody_instrument_index=None)
    melody_source_insts, _ = _auto_pick_with_intro(pm, info, melody_inst, sel)

    analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
    analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
    analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

    melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=MelodyConfig(quantize_to_beat=False))

    bpm = float(getattr(info, "tempo_bpm", 120.0) or 120.0)
    dur = float(getattr(info, "duration", 0.0) or 0.0)
    dur = max(dur, 1e-6)
    n_notes = len(melody_notes)
    notes_per_sec = n_notes / dur

    fast = bpm >= 150.0
    slow = bpm <= 85.0
    dense = notes_per_sec >= 3.0 or n_notes >= 450
    sparse = notes_per_sec <= 1.2 and n_notes <= 140

    ts = getattr(info, "time_signature", None)
    is_44 = True
    if ts is not None:
        is_44 = (getattr(ts, "numerator", 4) == 4) and (getattr(ts, "denominator", 4) == 4)

    # chords
    harmony_mode = "baseline (rules)" if (dense or fast) else "ml (transformer)"
    bars_per_chord = 2 if slow else 1

    chord_step_beats = 4.0 if slow else 2.0
    chord_include_key = True
    chord_stochastic = bool(sparse and not fast)

    chord_temp = 1.0
    chord_top_k = 12
    if sparse:
        chord_temp = 1.15
        chord_top_k = 18
    elif dense:
        chord_temp = 0.95
        chord_top_k = 10

    chord_repeat_penalty = 1.2
    chord_change_penalty = 0.15
    if sparse:
        chord_repeat_penalty = 1.25
        chord_change_penalty = 0.10
    elif dense:
        chord_repeat_penalty = 1.15
        chord_change_penalty = 0.22

    # drums
    if (not is_44) or dense or fast:
        drums_mode = "rules"
        ml_temp = 1.05
    else:
        drums_mode = "ml"
        ml_temp = 1.00 if not sparse else 1.05

    # melody preprocessing
    quantize_melody = bool(dense and bpm >= 110.0)

    # humanize
    humanize = True
    jitter_ms = 10.0 if fast else (18.0 if slow else 15.0)
    vel_jitter = 6 if fast else (10 if slow else 8)
    swing = 0.08 if fast else (0.18 if slow else 0.12)

    return {
        "harmony_mode": harmony_mode,
        "bars_per_chord": int(bars_per_chord),
        "chord_model_path": "data/ml/chord_model_new.pt",
        "chord_step_beats": float(chord_step_beats),
        "chord_include_key": bool(chord_include_key),
        "chord_stochastic": bool(chord_stochastic),
        "chord_temp": float(chord_temp),
        "chord_top_k": int(chord_top_k),
        "chord_repeat_penalty": float(chord_repeat_penalty),
        "chord_change_penalty": float(chord_change_penalty),
        "quantize_melody": bool(quantize_melody),
        "drums_mode": str(drums_mode),
        "ml_temp": float(ml_temp),
        "humanize": bool(humanize),
        "jitter_ms": float(jitter_ms),
        "vel_jitter": int(vel_jitter),
        "swing": float(swing),
        "auto_sections": True,
    }


def _apply_auto_settings(reco: Dict[str, Any], *, file_sig: str, force: bool = False) -> None:
    last_sig = st.session_state.get("_auto_sig")
    if force or (last_sig != file_sig):
        st.session_state["_auto_sig"] = file_sig
        for k, v in reco.items():
            st.session_state[k] = v


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI Backing Track Maker", layout="wide")

st.markdown(
    """
    <style>
      .card { border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
              padding: 16px 18px; background: rgba(255,255,255,0.04); }
      .muted { opacity: 0.75; }
      code { font-size: 0.95em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎼 AI Backing Track Maker")
st.caption("Upload a MIDI melody → generate a backing track (bass/pad/drums) → download a new multi-track MIDI.")

left, right = st.columns([0.42, 0.58], gap="large")

# Session defaults
DEFAULTS: Dict[str, Any] = {
    "mood_name": "neutral",
    "auto_settings": True,
    "harmony_mode": "baseline (rules)",
    "bars_per_chord": 1,
    "chord_model_path": "data/ml/chord_model_new.pt",
    "chord_step_beats": 2.0,
    "chord_include_key": True,
    "chord_stochastic": False,
    "chord_temp": 1.0,
    "chord_top_k": 12,
    "chord_repeat_penalty": 1.2,
    "chord_change_penalty": 0.15,
    "quantize_melody": False,
    "drums_mode": "rules",
    "ml_temp": 1.05,
    "humanize": True,
    "jitter_ms": 15.0,
    "vel_jitter": 8,
    "swing": 0.15,
    "use_seed": False,
    "seed_value": 0,
    "auto_sections": True,
    "make_bass": True,
    "make_pad": True,
    "make_drums": True,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Upload MIDI")
    uploaded = st.file_uploader("MIDI file (.mid / .midi)", type=["mid", "midi"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
    st.subheader("2) Controls")

    moods = list_moods()
    mood_name = st.selectbox(
        "Mood",
        moods,
        index=moods.index(st.session_state["mood_name"]) if st.session_state["mood_name"] in moods else 0,
        key="mood_name",
    )

    auto_settings = st.toggle(
        "Auto choose settings",
        value=bool(st.session_state["auto_settings"]),
        key="auto_settings",
        help="Picks good defaults for this MIDI. You can override anything in Advanced controls.",
    )

    if uploaded is not None and auto_settings:
        midi_bytes = uploaded.getvalue()
        file_sig = hashlib.sha1(midi_bytes).hexdigest()
        reco = recommend_settings(midi_bytes)

        c1, c2 = st.columns([0.62, 0.38])
        with c1:
            st.caption(
                f"Auto: chords={reco['harmony_mode']} · drums={reco['drums_mode']} · quantize={'on' if reco['quantize_melody'] else 'off'}"
            )
        with c2:
            reapply = st.button("Re-apply auto", use_container_width=True)

        _apply_auto_settings(reco, file_sig=file_sig, force=bool(reapply))

    # Simple surface controls
    auto_sections = st.toggle(
        "Auto song sections (intro/verse/chorus/outro)",
        value=bool(st.session_state["auto_sections"]),
        key="auto_sections",
    )
    structure_mode = "auto" if auto_sections else "none"

    st.markdown("**Backing tracks**")
    make_bass = st.toggle("Bass (rules)", value=bool(st.session_state["make_bass"]), key="make_bass")
    make_pad = st.toggle("Pad", value=bool(st.session_state["make_pad"]), key="make_pad")
    make_drums = st.toggle("Drums", value=bool(st.session_state["make_drums"]), key="make_drums")

    with st.expander("Advanced controls", expanded=False):
        st.markdown("**Harmony (chords)**")
        harmony_mode = st.selectbox(
            "Chord generator",
            ["baseline (rules)", "ml (transformer)"],
            index=["baseline (rules)", "ml (transformer)"].index(st.session_state["harmony_mode"])
            if st.session_state["harmony_mode"] in ["baseline (rules)", "ml (transformer)"]
            else 0,
            key="harmony_mode",
        )

        if str(harmony_mode).startswith("ml"):
            st.text_input("Chord model path", value=str(st.session_state["chord_model_path"]), key="chord_model_path")
            st.selectbox(
                "Chord step size (beats)",
                [1.0, 2.0, 4.0],
                index=[1.0, 2.0, 4.0].index(float(st.session_state["chord_step_beats"]))
                if float(st.session_state["chord_step_beats"]) in [1.0, 2.0, 4.0]
                else 1,
                key="chord_step_beats",
            )
            st.toggle("Include key features (recommended)", value=bool(st.session_state["chord_include_key"]), key="chord_include_key")
            st.toggle("Stochastic chords (more variety)", value=bool(st.session_state["chord_stochastic"]), key="chord_stochastic")
            st.slider("Chord temperature", 0.7, 1.6, float(st.session_state["chord_temp"]), 0.01, key="chord_temp")
            st.slider("Chord top-k (0 = no top-k)", 0, 40, int(st.session_state["chord_top_k"]), 1, key="chord_top_k")
            st.slider("Chord repeat penalty", 0.0, 3.0, float(st.session_state["chord_repeat_penalty"]), 0.05, key="chord_repeat_penalty")
            st.slider(
                "Chord smoothness (change penalty)",
                0.0,
                0.6,
                float(st.session_state["chord_change_penalty"]),
                0.01,
                key="chord_change_penalty",
                disabled=bool(st.session_state["chord_stochastic"]),
            )
            st.slider("Bars per chord (baseline only)", 1, 4, int(st.session_state["bars_per_chord"]), 1, key="bars_per_chord", disabled=True)
        else:
            st.slider("Bars per chord", 1, 4, int(st.session_state["bars_per_chord"]), 1, key="bars_per_chord")

        st.markdown("**Melody preprocessing**")
        st.toggle("Quantize melody to beat grid", value=bool(st.session_state["quantize_melody"]), key="quantize_melody")

        st.divider()
        st.markdown("**Drums**")
        st.selectbox(
            "Drums generator",
            ["rules", "ml"],
            index=["rules", "ml"].index(st.session_state["drums_mode"]) if st.session_state["drums_mode"] in ["rules", "ml"] else 0,
            key="drums_mode",
        )
        st.slider(
            "ML drum temperature",
            0.8,
            1.4,
            float(st.session_state["ml_temp"]),
            0.01,
            key="ml_temp",
            disabled=(st.session_state["drums_mode"] != "ml"),
        )

        st.divider()
        st.markdown("**Humanize**")
        st.toggle("Humanize timing/velocity", value=bool(st.session_state["humanize"]), key="humanize")
        st.slider("Timing jitter (ms)", 0.0, 35.0, float(st.session_state["jitter_ms"]), 1.0, key="jitter_ms", disabled=not bool(st.session_state["humanize"]))
        st.slider("Velocity jitter", 0, 20, int(st.session_state["vel_jitter"]), 1, key="vel_jitter", disabled=not bool(st.session_state["humanize"]))
        st.slider("Swing (0..1)", 0.0, 0.6, float(st.session_state["swing"]), 0.01, key="swing", disabled=not bool(st.session_state["humanize"]))

        st.divider()
        st.markdown("**Seed**")
        st.number_input("Seed value", value=int(st.session_state["seed_value"]), step=1, key="seed_value")
        st.toggle("Use seed", value=bool(st.session_state["use_seed"]), key="use_seed")

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
                if not pm.instruments[i].is_drum:
                    valid.append(i)
        if not valid:
            raise RuntimeError("No valid (non-drum) melody instruments selected.")
        melody_source_insts = [pm.instruments[i] for i in valid]
        used_melody_indices = valid
    else:
        melody_source_insts, picked_intro_idxs = _auto_pick_with_intro(pm, info, melody_inst, sel)
        used_melody_indices = [sel.instrument_index] + picked_intro_idxs

    # Combine selected lead instruments for analysis
    analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
    analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
    analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

    melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=MelodyConfig(quantize_to_beat=quantize_melody))
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

    # Arrange (rules bass only)
    arrangement = arrange_backing(
        chords=chords,
        grid=grid,
        mood=mood,
        make_bass=bool(make_bass),
        make_pad=make_pad,
        make_drums=make_drums,
        seed=seed,
        structure_mode=structure_mode,
        drums_mode=drums_mode,
        ml_drums_model_path="data/ml/drum_model.pt",
        ml_drums_temperature=ml_temp,
    )

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
        "bass_mode": "rules",
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
            melody_track_indices = [int(x.split(':')[0].strip()) for x in picked] if picked else None

        with st.expander("Show instrument list"):
            st.json(
                [
                    {"idx": i, "name": (inst.name or f"Instrument {i}"), "is_drum": inst.is_drum, "notes": len(inst.notes)}
                    for i, inst in enumerate(pm_preview.instruments)
                ]
            )

        st.markdown("</div>", unsafe_allow_html=True)


if generate_btn and uploaded is not None:
    # Pull values from session_state (single source of truth)
    seed: Optional[int] = int(st.session_state["seed_value"]) if bool(st.session_state["use_seed"]) else None
    structure_mode = "auto" if bool(st.session_state["auto_sections"]) else "none"

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Output")

        try:
            with st.spinner("Generating backing track..."):
                out_path, meta = run_pipeline(
                    midi_bytes=uploaded.getvalue(),
                    mood_name=st.session_state["mood_name"],
                    harmony_mode=st.session_state["harmony_mode"],
                    chord_model_path=st.session_state["chord_model_path"],
                    chord_step_beats=float(st.session_state["chord_step_beats"]),
                    chord_include_key=bool(st.session_state["chord_include_key"]),
                    chord_stochastic=bool(st.session_state["chord_stochastic"]),
                    chord_temp=float(st.session_state["chord_temp"]),
                    chord_top_k=int(st.session_state["chord_top_k"]),
                    chord_repeat_penalty=float(st.session_state["chord_repeat_penalty"]),
                    chord_change_penalty=float(st.session_state["chord_change_penalty"]),
                    bars_per_chord=int(st.session_state["bars_per_chord"]),
                    quantize_melody=bool(st.session_state["quantize_melody"]),
                    make_bass=bool(st.session_state["make_bass"]),
                    make_pad=bool(st.session_state["make_pad"]),
                    make_drums=bool(st.session_state["make_drums"]),
                    melody_track_indices=melody_track_indices,
                    seed=seed,
                    structure_mode=structure_mode,
                    drums_mode=st.session_state["drums_mode"],
                    ml_temp=float(st.session_state["ml_temp"]),
                    humanize=bool(st.session_state["humanize"]),
                    jitter_ms=float(st.session_state["jitter_ms"]),
                    vel_jitter=int(st.session_state["vel_jitter"]),
                    swing=float(st.session_state["swing"]),
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

        st.markdown("**Bass:** `rules`")

        midi_out_bytes = out_path.read_bytes()
        st.download_button(
            label="⬇️ Download generated MIDI",
            data=midi_out_bytes,
            file_name="backing_track.mid",
            mime="audio/midi",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
