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
# GM Instrument presets
# ----------------------------
PAD_PRESETS = [
    ("Electric Piano 1", 4),
    ("Electric Piano 2", 5),
    ("Acoustic Grand Piano", 0),
    ("Acoustic Guitar (nylon)", 24),
    ("Acoustic Guitar (steel)", 25),
    ("Electric Guitar (clean)", 27),
    ("Strings Ensemble 1", 48),
    ("Choir Aahs", 52),
    ("Brass Section", 61),
    ("Synth Pad 2 (warm)", 89),
    ("Synth Pad 1 (new age)", 88),
]

BASS_PRESETS = [
    ("Electric Bass (finger)", 33),
    ("Electric Bass (pick)", 34),
    ("Acoustic Bass", 32),
    ("Synth Bass 1", 38),
    ("Synth Bass 2", 39),
]


def _preset_index(presets: list[tuple[str, int]], program: int) -> int:
    for i, (_, p) in enumerate(presets):
        if int(p) == int(program):
            return i
    return 0


# ----------------------------
# Auto-pick helpers
# ----------------------------
def _median_pitch(inst: pretty_midi.Instrument) -> float:
    pitches = sorted(n.pitch for n in inst.notes)
    if not pitches:
        return 0.0
    m = len(pitches)
    return float(pitches[m // 2]) if (m % 2 == 1) else 0.5 * (pitches[m // 2 - 1] + pitches[m // 2])


def _auto_pick_with_intro(
    pm: pretty_midi.PrettyMIDI,
    info_or_sel,
    melody_inst: Optional[pretty_midi.Instrument] = None,
    sel=None,
    max_intro: int = 2,
) -> tuple[list[pretty_midi.Instrument], list[int]]:
    """
    Backwards compatible:
      - _auto_pick_with_intro(pm, sel)
      - _auto_pick_with_intro(pm, info, melody_inst, sel)
      - _auto_pick_with_intro(pm, info, sel)   (rare, but supported)

    Also supports multi-lead selection via sel.instrument_indices.
    Returns:
      (melody_source_insts, picked_intro_idxs)
    """
    info = None

    # Case A: called as (pm, sel)
    if sel is None and melody_inst is None and hasattr(info_or_sel, "instrument_index"):
        sel = info_or_sel

    # Case B: called as (pm, info, sel) where 3rd arg is actually sel
    elif sel is None and melody_inst is not None and hasattr(melody_inst, "instrument_index") and not hasattr(melody_inst, "notes"):
        info = info_or_sel
        sel = melody_inst
        melody_inst = None

    # Case C: called as (pm, info, melody_inst, sel)
    else:
        info = info_or_sel

    if sel is None:
        raise ValueError("_auto_pick_with_intro: could not determine `sel` (melody selection).")

    # Base lead indices: multi-select if available, else fallback to single
    base_idxs = list(getattr(sel, "instrument_indices", None) or [int(getattr(sel, "instrument_index", 0))])

    # Sanitize indices (in-range, not drums)
    valid_base: list[int] = []
    for i in base_idxs:
        i = int(i)
        if 0 <= i < len(pm.instruments) and (not pm.instruments[i].is_drum) and pm.instruments[i].notes:
            if i not in valid_base:
                valid_base.append(i)

    if not valid_base:
        # ultimate fallback: first non-drum with notes
        for i, inst in enumerate(pm.instruments):
            if (not inst.is_drum) and inst.notes:
                valid_base = [i]
                break

    if melody_inst is None:
        melody_inst = pm.instruments[valid_base[0]]

    # Use info.duration if available, else end_time
    song_end = float(getattr(info, "duration", 0.0) or 0.0)
    if song_end <= 1e-6:
        song_end = float(pm.get_end_time())

    # Median pitch of the combined base lead(s)
    all_base_pitches: list[int] = []
    for i in valid_base:
        all_base_pitches.extend([int(n.pitch) for n in pm.instruments[i].notes])
    if all_base_pitches:
        all_base_pitches.sort()
        m = len(all_base_pitches)
        main_med = float(all_base_pitches[m // 2]) if (m % 2 == 1) else 0.5 * (all_base_pitches[m // 2 - 1] + all_base_pitches[m // 2])
    else:
        main_med = _median_pitch(melody_inst)

    # Find short, high-pitch intro candidates (excluding *all* base leads)
    base_set = set(valid_base)
    intro_candidates: list[tuple[int, float, int]] = []  # (idx, median_pitch, note_count)

    for idx, inst in enumerate(pm.instruments):
        if idx in base_set:
            continue
        if inst.is_drum or not inst.notes:
            continue

        start = float(min(n.start for n in inst.notes))
        end = float(max(n.end for n in inst.notes))
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
    picked_intro_idxs = [idx for (idx, _, _) in intro_candidates[: max(0, int(max_intro))]]

    # Used indices: intro first, then ALL base leads
    used_indices: list[int] = []
    for i in picked_intro_idxs + valid_base:
        if i not in used_indices:
            used_indices.append(i)

    melody_source_insts = [pm.instruments[i] for i in used_indices]
    return melody_source_insts, picked_intro_idxs

# ----------------------------
# Auto settings (kept)
# ----------------------------
@st.cache_data(show_spinner=False)
def recommend_settings(midi_bytes: bytes) -> Dict[str, Any]:
    """
    Heuristic 'auto' settings.
    Project preference:
      - Chords default to RULES
      - Drums default to ML

    NOTE: We intentionally do NOT set pad/bass program here so user selection persists.
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

    harmony_mode = "baseline (rules)"
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

    drums_mode = "ml"
    if fast or sparse:
        ml_temp = 1.05
    else:
        ml_temp = 1.00

    quantize_melody = bool(dense and bpm >= 110.0)

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

DEFAULTS: Dict[str, Any] = {
    "mood_name": "neutral",
    "auto_settings": True,
    "harmony_mode": "baseline (rules)",   # chords default = rules
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
    "drums_mode": "ml",                  # drums default = ML
    "ml_temp": 1.00,
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
    # NEW: instrument programs
    "pad_program": 4,    # Electric Piano 1
    "bass_program": 33,  # Electric Bass (finger)
    "pad_custom_on": False,
    "bass_custom_on": False,
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
    st.selectbox(
        "Mood",
        moods,
        index=moods.index(st.session_state["mood_name"]) if st.session_state["mood_name"] in moods else 0,
        key="mood_name",
    )

    st.toggle(
        "Auto choose settings",
        value=bool(st.session_state["auto_settings"]),
        key="auto_settings",
        help="Picks good defaults for this MIDI. You can override anything in Advanced controls.",
    )

    if uploaded is not None and st.session_state["auto_settings"]:
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

    st.toggle(
        "Auto song sections (intro/verse/chorus/outro)",
        value=bool(st.session_state["auto_sections"]),
        key="auto_sections",
    )
    structure_mode = "auto" if bool(st.session_state["auto_sections"]) else "none"

    st.markdown("**Backing tracks**")
    st.toggle("Bass (rules)", value=bool(st.session_state["make_bass"]), key="make_bass")
    st.toggle("Pad", value=bool(st.session_state["make_pad"]), key="make_pad")
    st.toggle("Drums", value=bool(st.session_state["make_drums"]), key="make_drums")

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
            index=["rules", "ml"].index(st.session_state["drums_mode"]) if st.session_state["drums_mode"] in ["rules", "ml"] else 1,
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

        # NEW: instrument programs
        st.divider()
        st.markdown("**Instruments (MIDI / General MIDI programs)**")
        st.caption("This changes the instrument program in the output MIDI. The exact sound depends on your MIDI synth/DAW.")

        # Pad selection
        pad_idx = _preset_index(PAD_PRESETS, int(st.session_state["pad_program"]))
        pad_preset = st.selectbox(
            "Chords instrument (Pad track)",
            PAD_PRESETS,
            index=pad_idx,
            format_func=lambda x: f"{x[0]} ({x[1]})",
            key="pad_preset",
        )
        st.toggle("Custom pad program # (0-127)", value=bool(st.session_state["pad_custom_on"]), key="pad_custom_on")
        if st.session_state["pad_custom_on"]:
            st.number_input("Pad program", 0, 127, int(st.session_state["pad_program"]), key="pad_program")
        else:
            st.session_state["pad_program"] = int(pad_preset[1])

        # Bass selection
        bass_idx = _preset_index(BASS_PRESETS, int(st.session_state["bass_program"]))
        bass_preset = st.selectbox(
            "Bass instrument (Bass track)",
            BASS_PRESETS,
            index=bass_idx,
            format_func=lambda x: f"{x[0]} ({x[1]})",
            key="bass_preset",
        )
        st.toggle("Custom bass program # (0-127)", value=bool(st.session_state["bass_custom_on"]), key="bass_custom_on")
        if st.session_state["bass_custom_on"]:
            st.number_input("Bass program", 0, 127, int(st.session_state["bass_program"]), key="bass_program")
        else:
            st.session_state["bass_program"] = int(bass_preset[1])

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
    pad_program: int,
    bass_program: int,
) -> tuple[Path, dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_bytes)
        in_path = Path(f.name)

    # If the user manually picked tracks, we still let that override auto
    first_idx = melody_track_indices[0] if melody_track_indices else None
    pm, info, grid, melody_inst, sel = load_and_prepare(in_path, melody_instrument_index=first_idx)

    picked_intro_idxs: list[int] = []
    used_melody_indices: list[int] = []

    # Decide which instruments are the "lead"
    if melody_track_indices:
        valid: list[int] = []
        for i in melody_track_indices:
            if 0 <= i < len(pm.instruments) and not pm.instruments[i].is_drum:
                valid.append(i)
        if not valid:
            raise RuntimeError("No valid (non-drum) melody instruments selected.")
        melody_source_insts = [pm.instruments[i] for i in valid]
        used_melody_indices = valid
    else:
        # NEW: use multi-lead selection from midi_io (fallback to single)
        base_idxs = getattr(sel, "instrument_indices", None)
        if not base_idxs:
            base_idxs = [getattr(sel, "instrument_index", 0)]

        base_valid: list[int] = []
        for i in base_idxs:
            try:
                ii = int(i)
            except Exception:
                continue
            if 0 <= ii < len(pm.instruments) and not pm.instruments[ii].is_drum:
                base_valid.append(ii)

        # hard fallback if something is off
        if not base_valid:
            fb = int(getattr(sel, "instrument_index", 0))
            if 0 <= fb < len(pm.instruments) and not pm.instruments[fb].is_drum:
                base_valid = [fb]

        # Keep your intro heuristic, but don't let it "steal" one of the base leads
        _, picked_intro_idxs = _auto_pick_with_intro(pm, info, melody_inst, sel)
        picked_intro_idxs = [i for i in picked_intro_idxs if i not in base_valid]

        used_melody_indices = []
        for i in picked_intro_idxs + base_valid:
            if i not in used_melody_indices:
                used_melody_indices.append(i)

        melody_source_insts = [pm.instruments[i] for i in used_melody_indices]

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

    # Arrange
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

    render_cfg = RenderConfig(
        melody_program=int(melody_source_insts[0].program),
        bass_program=int(bass_program),
        pad_program=int(pad_program),
    )

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
        "pad_program": int(pad_program),
        "bass_program": int(bass_program),
        "melody_program": int(melody_source_insts[0].program) if melody_source_insts else None,
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
                    pad_program=int(st.session_state["pad_program"]),
                    bass_program=int(st.session_state["bass_program"]),
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

        st.markdown(f"**Pad program:** `{meta['pad_program']}`  ·  **Bass program:** `{meta['bass_program']}`")

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

        midi_out_bytes = out_path.read_bytes()
        st.download_button(
            label="⬇️ Download generated MIDI",
            data=midi_out_bytes,
            file_name="backing_track.mid",
            mime="audio/midi",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
