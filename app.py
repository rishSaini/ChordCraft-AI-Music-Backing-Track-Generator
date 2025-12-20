# app.py
from __future__ import annotations

from backingtrack.ml_harmony.infer import predict_chords_ml

import tempfile
from pathlib import Path
from typing import Optional

import pretty_midi
import streamlit as st

from backingtrack.arrange import arrange_backing
from backingtrack.harmony_baseline import generate_chords
from backingtrack.humanize import HumanizeConfig, humanize_arrangement
from backingtrack.key_detect import estimate_key, key_to_string
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.midi_io import load_and_prepare
from backingtrack.moods import apply_mood_to_key, get_mood, list_moods
from backingtrack.render import RenderConfig, write_midi
from backingtrack.types import Note

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
# NEW: match cli.py auto-intro logic
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
    Use Fix 2's auto-picked melody_inst as the main lead, but also add
    a short, high-pitch intro lead track if it exists (like st.mid).
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


st.set_page_config(
    page_title="AI Backing Track Maker",
    page_icon="🎼",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1200px; }
      .stButton>button {
        border-radius: 14px;
        padding: 0.75rem 1.1rem;
        font-weight: 650;
      }
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.04);
      }
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
    mood_name = st.selectbox(
        "Mood",
        moods,
        index=moods.index("neutral") if "neutral" in moods else 0,
    )

    bars_per_chord = st.slider("Bars per chord", min_value=1, max_value=4, value=1, step=1)
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

    drums_mode = st.selectbox("Drums", ["rules", "ml"], index=0)
    ml_temp = st.slider("ML drum temperature", 0.8, 1.4, 1.05, 0.01)

    harmony_mode = st.selectbox("Harmony (chords)", ["baseline", "ml"], index=0)

    ml_chord_model_path = "data/ml/chord_model.pt"
    ml_chord_change_penalty = 0.6
    ml_chord_include_key = True

    if harmony_mode == "ml":
        ml_chord_model_path = st.text_input("Chord model path", value="data/ml/chord_model.pt")
        ml_chord_change_penalty = st.slider("Chord smoothing (change penalty)", 0.0, 2.0, 0.6, 0.05)
        ml_chord_include_key = st.toggle("Chord model uses key features", value=True)


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
    harmony_mode: str,
    ml_chord_model_path: str,
    ml_chord_change_penalty: float,
    ml_chord_include_key: bool,
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
        # resolve indices we used (in order)
        for i, inst in enumerate(pm.instruments):
            if any(inst is x for x in melody_source_insts):
                used_melody_indices.append(i)

    # Combine selected lead instruments for analysis (key/chords)
    analysis_inst = pretty_midi.Instrument(
        program=int(melody_source_insts[0].program),
        is_drum=False,
        name="Analysis",
    )
    analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
    analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

    mel_cfg = MelodyConfig(quantize_to_beat=quantize_melody)
    melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=mel_cfg)
    if not melody_notes:
        raise RuntimeError("No melody notes extracted. Try selecting a different melody track (or multiple tracks).")

    mood = get_mood(mood_name)
    raw_key = estimate_key(melody_notes)
    key = apply_mood_to_key(raw_key, mood)

    if harmony_mode == "ml":
        chords = predict_chords_ml(
            melody_notes=melody_notes,
            grid=grid,
            duration_seconds=info.duration,
            model_path=ml_chord_model_path,
            include_key=ml_chord_include_key,
            change_penalty=ml_chord_change_penalty,
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


    arrangement = arrange_backing(
        chords=chords,
        grid=grid,
        mood=mood,
        make_bass=make_bass,
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
            HumanizeConfig(
                timing_jitter_ms=jitter_ms,
                velocity_jitter=vel_jitter,
                swing=swing,
                seed=seed,
            ),
        )

    out_path = Path(tempfile.mkstemp(suffix=".mid")[1])

    render_cfg = RenderConfig(melody_program=int(melody_source_insts[0].program))

    # Render the lead from original instrument(s)
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

    pm_preview, info_preview, grid_preview, melody_inst_preview, sel_preview = load_and_prepare(
        tmp_path, melody_instrument_index=None
    )

    # NEW: show which intro tracks auto-mode will include
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
                    bars_per_chord=bars_per_chord,
                    quantize_melody=quantize_melody,
                    make_bass=make_bass,
                    make_pad=make_pad,
                    make_drums=make_drums,
                    melody_track_indices=melody_track_indices,
                    seed=seed,
                    structure_mode=structure_mode,
                    drums_mode=drums_mode,
                    ml_temp=ml_temp,
                    humanize=humanize,
                    jitter_ms=jitter_ms,
                    vel_jitter=int(vel_jitter),
                    swing=float(swing),
                    harmony_mode=harmony_mode,
                    ml_chord_model_path=ml_chord_model_path,
                    ml_chord_change_penalty=ml_chord_change_penalty,
                    ml_chord_include_key=ml_chord_include_key,

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

        midi_out_bytes = out_path.read_bytes()
        st.download_button(
            label="⬇️ Download generated MIDI",
            data=midi_out_bytes,
            file_name="backing_track.mid",
            mime="audio/midi",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
