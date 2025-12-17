# src/backingtrack/render.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pretty_midi

from .types import MidiInfo, Note, TimeSignature
from .arrange import Arrangement, TrackName


# Reasonable GM programs (0-127). These are General MIDI instrument programs:
# 0 = Acoustic Grand Piano, 33 = Electric Bass (finger), 48 = Strings Ensemble 1
DEFAULT_PROGRAMS: Dict[str, int] = {
    "melody": 0,
    "bass": 33,
    "pad": 48,
}

DEFAULT_NAMES: Dict[str, str] = {
    "melody": "Melody",
    "bass": "Bass",
    "pad": "Pad",
    "drums": "Drums",
}


@dataclass(frozen=True)
class RenderConfig:
    """
    Controls what instruments/programs we use in the output MIDI.
    """
    melody_program: int = DEFAULT_PROGRAMS["melody"]
    bass_program: int = DEFAULT_PROGRAMS["bass"]
    pad_program: int = DEFAULT_PROGRAMS["pad"]

    # If True, we write tempo + time signature meta events at time 0.
    write_metadata: bool = True

    # If provided, use this as the output tempo (otherwise use MidiInfo.tempo_bpm).
    override_tempo_bpm: Optional[float] = None


def _to_pretty_midi_note(n: Note) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=int(n.velocity),
        pitch=int(n.pitch),
        start=float(n.start),
        end=float(n.end),
    )


def _notes_to_instrument(
    notes: Sequence[Note],
    *,
    program: int,
    name: str,
    is_drum: bool = False,
) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=int(program), is_drum=bool(is_drum), name=name)
    inst.notes = [_to_pretty_midi_note(n) for n in notes]
    # Keep notes sorted (nice for debugging + some MIDI players)
    inst.notes.sort(key=lambda x: (x.start, x.pitch))
    return inst


def build_pretty_midi(
    melody_notes: Sequence[Note],
    arrangement: Arrangement,
    info: MidiInfo,
    config: RenderConfig = RenderConfig(),
) -> pretty_midi.PrettyMIDI:
    """
    Build a PrettyMIDI object containing:
      - Melody track
      - Bass track (if present)
      - Pad track (if present)
      - Drum track (if present)

    This does NOT write to disk. Use write_midi(...) for that.
    """
    tempo = float(config.override_tempo_bpm) if config.override_tempo_bpm is not None else float(info.tempo_bpm)

    # PrettyMIDI lets you set an initial tempo at construction time.
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Write time signature metadata (optional)
    if config.write_metadata:
        ts: TimeSignature = info.time_signature
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(int(ts.numerator), int(ts.denominator), time=0.0)
        )
        # tempo is already set via initial_tempo

    # Melody
    if melody_notes:
        melody_inst = _notes_to_instrument(
            melody_notes,
            program=config.melody_program,
            name=DEFAULT_NAMES["melody"],
            is_drum=False,
        )
        pm.instruments.append(melody_inst)

    # Backing tracks
    tracks: Dict[TrackName, Sequence[Note]] = arrangement.tracks

    bass_notes = tracks.get("bass", [])
    if bass_notes:
        pm.instruments.append(
            _notes_to_instrument(
                bass_notes,
                program=config.bass_program,
                name=DEFAULT_NAMES["bass"],
                is_drum=False,
            )
        )

    pad_notes = tracks.get("pad", [])
    if pad_notes:
        pm.instruments.append(
            _notes_to_instrument(
                pad_notes,
                program=config.pad_program,
                name=DEFAULT_NAMES["pad"],
                is_drum=False,
            )
        )

    drum_notes = tracks.get("drums", [])
    if drum_notes:
        # For drums, program is ignored by most players; must set is_drum=True.
        pm.instruments.append(
            _notes_to_instrument(
                drum_notes,
                program=0,
                name=DEFAULT_NAMES["drums"],
                is_drum=True,
            )
        )

    return pm


def write_midi(
    output_path: str | Path,
    melody_notes: Sequence[Note],
    arrangement: Arrangement,
    info: MidiInfo,
    config: RenderConfig = RenderConfig(),
) -> Path:
    """
    Build and write the MIDI file to disk.
    Returns the output path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pm = build_pretty_midi(melody_notes, arrangement, info, config=config)
    pm.write(str(out))
    return out
