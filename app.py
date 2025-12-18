# app.py
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from backingtrack.midi_io import load_and_prepare
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.key_detect import estimate_key, key_to_string
from backingtrack.moods import get_mood, list_moods, apply_mood_to_key
from backingtrack.harmony_baseline import generate_chords
from backingtrack.arrange import arrange_backing
from backingtrack.render import write_midi
from backingtrack.humanize import HumanizeConfig, humanize_arrangement

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_label(root_pc: int, quality: str, extensions: tuple[int, ...]) -> str:
    name = f"{PC_NAMES[root_pc % 12]}{'' if quality == 'maj' else quality}"
    # minimal readable extensions
    if 10 in extensions:
        name += "7"
    elif 11 in extensions:
        name += "maj7"
    if 14 in extensions:
        name += "add9"
    return name


st.set_page_config(
    page_title="AI Backing Track Maker",
    page_icon="🎼",
    layout="wide",
)

# --- Simple styling to look more “product-y” ---
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


# --- Layout ---
left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Upload MIDI")

    uploaded = st.file_uploader("MIDI file (.mid / .midi)", type=["mid", "midi"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
    st.subheader("2) Controls")

    mood_name = st.selectbox("Mood", list_moods(), index=list_moods().index("neutral") if "neutral" in list_moods() else 0)

    bars_per_chord = st.slider("Bars per chord", min_value=1, max_value=4, value=1, step=1)
    quantize_melody = st.toggle("Quantize melody to beat grid", value=False)

    st.divider()
    st.markdown("**Humanize**")
    humanize = st.toggle("Humanize timing/velocity", value=True)
    jitter_ms = st.slider("Timing jitter (ms)", 0.0, 35.0, 15.0, 1.0)
    vel_jitter = st.slider("Velocity jitter", 0, 20, 8, 1)
    swing = st.slider("Swing (0..1)", 0.0, 0.6, 0.15, 0.01)
    seed = st.number_input("Seed (optional)", value=0, step=1)
    use_seed = st.toggle("Use seed", value=False)

    structure_mode = "auto" if st.toggle("Auto song sections (intro/verse/chorus/outro)", value=True) else "none"

    st.divider()
    st.markdown("**Backing tracks**")
    make_bass = st.toggle("Bass", value=True)
    make_pad = st.toggle("Pad", value=True)
    make_drums = st.toggle("Drums", value=True)

    drums_mode = st.selectbox("Drums", ["rules", "ml"], index=0)
    ml_temp = st.slider("ML drum temperature", 0.8, 1.4, 1.05, 0.01)

    st.markdown("</div>", unsafe_allow_html=True)

    generate_btn = st.button("✨ Generate backing track", use_container_width=True, disabled=(uploaded is None))


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output")
    st.markdown('<p class="muted">Once generated, you’ll see detected key, chosen melody track, chord progression preview, and a download button.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def run_pipeline(midi_bytes: bytes, mood_name: str, bars_per_chord: int, quantize_melody: bool,
                 make_bass: bool, make_pad: bool, make_drums: bool,
                 melody_track_index: Optional[int]) -> tuple[Path, dict]:
    # Save upload to a real temp .mid because our loader expects a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_bytes)
        in_path = Path(f.name)

    pm, info, grid, melody_inst, sel = load_and_prepare(in_path, melody_instrument_index=melody_track_index)

    mel_cfg = MelodyConfig(quantize_to_beat=quantize_melody)
    melody_notes = extract_melody_notes(melody_inst, grid=grid, config=mel_cfg)

    mood = get_mood(mood_name)
    raw_key = estimate_key(melody_notes)
    key = apply_mood_to_key(raw_key, mood)

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
                seed=int(seed) if use_seed else None,
            ),
        )


    out_path = Path(tempfile.mkstemp(suffix=".mid")[1])
    write_midi(out_path, melody_notes, arrangement, info)

    meta = {
        "info": info,
        "selection": sel,
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


# If a file is uploaded, show instrument selection (auto or pick a track)
melody_track_index: Optional[int] = None

if uploaded is not None:
    # write bytes to temp just to read instruments list quickly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(uploaded.getvalue())
        tmp_path = Path(f.name)

    pm_preview, info_preview, grid_preview, melody_inst_preview, sel_preview = load_and_prepare(tmp_path, melody_instrument_index=None)

    with left:
        st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
        st.subheader("3) Melody track")

        # Build options
        inst_options = ["Auto (recommended)"]
        for i, inst in enumerate(pm_preview.instruments):
            nm = inst.name or f"Instrument {i}"
            tag = "DRUMS" if inst.is_drum else "INST"
            inst_options.append(f"{i}: {nm}  ·  {tag}  ·  notes={len(inst.notes)}")

        choice = st.selectbox("Choose melody instrument", inst_options, index=0)
        if choice != "Auto (recommended)":
            melody_track_index = int(choice.split(":")[0].strip())

        st.caption(f"Auto-picked track (preview): idx={sel_preview.instrument_index}, name='{sel_preview.instrument_name}'")
        st.markdown("</div>", unsafe_allow_html=True)


if generate_btn and uploaded is not None:
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Output")

        with st.spinner("Generating backing track..."):
            out_path, meta = run_pipeline(
                midi_bytes=uploaded.getvalue(),
                mood_name=mood_name,
                bars_per_chord=bars_per_chord,
                quantize_melody=quantize_melody,
                make_bass=make_bass,
                make_pad=make_pad,
                make_drums=make_drums,
                melody_track_index=melody_track_index,
            )

        info = meta["info"]
        sel = meta["selection"]

        st.success("Done ✅")

        # Summary
        colA, colB, colC = st.columns(3)
        colA.metric("Tempo (BPM)", f"{info.tempo_bpm:.1f}")
        colB.metric("Time Signature", f"{info.time_signature.numerator}/{info.time_signature.denominator}")
        colC.metric("Duration (s)", f"{info.duration:.1f}")

        st.markdown(f"**Melody track:** idx={sel.instrument_index} · `{sel.instrument_name}`")
        st.markdown(f"**Melody notes extracted:** `{meta['melody_note_count']}`")

        st.markdown(f"**Detected key:** {key_to_string(meta['raw_key'])}")
        if meta["key"] != meta["raw_key"]:
            st.markdown(f"**After mood '{meta['mood'].name}' bias:** {key_to_string(meta['key'])}")

        st.markdown("**Backing note counts:**")
        st.json(meta["arrangement_counts"])

        # Chord preview (first 8 chords)
        chords = meta["chords"]
        preview = " · ".join(chord_label(c.root_pc, c.quality, c.extensions) for c in chords[:8])
        st.markdown("**Chord progression preview:**")
        st.code(preview if preview else "(none)")

        # Download
        midi_bytes = out_path.read_bytes()
        st.download_button(
            label="⬇️ Download generated MIDI",
            data=midi_bytes,
            file_name="backing_track.mid",
            mime="audio/midi",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
