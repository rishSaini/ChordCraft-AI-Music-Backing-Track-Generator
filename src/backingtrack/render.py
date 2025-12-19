# src/backingtrack/render.py
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pretty_midi

from .arrange import Arrangement, TrackName
from .types import MidiInfo, Note, TimeSignature

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

    write_metadata: bool = True
    override_tempo_bpm: Optional[float] = None

    # ----------------------------
    # Backing mix controls (NEW)
    # ----------------------------
    auto_balance_backing: bool = True          # boost backing relative to melody if melody is very loud
    backing_target_ratio: float = 0.90         # aim backing median ~= 0.90 * melody median
    max_backing_boost: float = 1.80            # clamp auto boost so backing doesn't slam to 127

    backing_vel_scale: float = 1.00            # manual global backing gain (multiplies on top of auto)
    bass_vel_scale: float = 1.00               # per-track multipliers
    pad_vel_scale: float = 1.15
    drums_vel_scale: float = 1.10


def _to_pretty_midi_note(n: Note, vel_scale: float = 1.0) -> pretty_midi.Note:
    v = int(max(1, min(127, round(int(n.velocity) * vel_scale))))
    return pretty_midi.Note(
        velocity=v,
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
    vel_scale: float = 1.0,
) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=int(program), is_drum=bool(is_drum), name=name)
    inst.notes = [_to_pretty_midi_note(n, vel_scale=vel_scale) for n in notes]
    inst.notes.sort(key=lambda x: (x.start, x.pitch))
    return inst


# ----------------------------
# Velocity stats helpers (NEW)
# ----------------------------
def _median_int(xs: Sequence[int], default: int = 100) -> int:
    if not xs:
        return int(default)
    ys = sorted(int(x) for x in xs)
    return int(ys[len(ys) // 2])


def _median_velocity_notes(notes: Sequence[Note], default: int = 100) -> int:
    return _median_int([int(n.velocity) for n in notes], default=default)


def _median_velocity_insts(insts: Sequence[pretty_midi.Instrument], default: int = 100) -> int:
    vels: list[int] = []
    for inst in insts:
        for n in inst.notes:
            vels.append(int(n.velocity))
    return _median_int(vels, default=default)


def build_pretty_midi(
    melody_notes: Sequence[Note],
    arrangement: Arrangement,
    info: MidiInfo,
    config: RenderConfig = RenderConfig(),
    melody_source_insts: Optional[Sequence[pretty_midi.Instrument]] = None,
) -> pretty_midi.PrettyMIDI:
    tempo = float(config.override_tempo_bpm) if config.override_tempo_bpm is not None else float(info.tempo_bpm)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    if config.write_metadata:
        ts: TimeSignature = info.time_signature
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(int(ts.numerator), int(ts.denominator), time=0.0)
        )

    # Melody (preferred: deep-copy original instrument(s) so pedal/CC/pitch bends are preserved)
    if melody_source_insts:
        for j, src in enumerate(melody_source_insts):
            lead = deepcopy(src)
            lead.is_drum = False
            lead.name = DEFAULT_NAMES["melody"] if len(melody_source_insts) == 1 else f"Melody {j + 1}"

            # Make event ordering deterministic + player-safe
            lead.notes.sort(key=lambda x: (x.start, x.pitch))
            lead.control_changes.sort(key=lambda cc: cc.time)
            lead.pitch_bends.sort(key=lambda pb: pb.time)

            pm.instruments.append(lead)

    elif melody_notes:
        pm.instruments.append(
            _notes_to_instrument(
                melody_notes,
                program=config.melody_program,
                name=DEFAULT_NAMES["melody"],
                is_drum=False,
            )
        )

    tracks: Dict[TrackName, Sequence[Note]] = arrangement.tracks

    bass_notes = tracks.get("bass", [])
    pad_notes = tracks.get("pad", [])
    drum_notes = tracks.get("drums", [])

    # ----------------------------
    # Auto-balance backing vs melody (NEW)
    # ----------------------------
    backing_notes: list[Note] = []
    backing_notes += list(bass_notes)
    backing_notes += list(pad_notes)
    backing_notes += list(drum_notes)

    auto_boost = 1.0
    if config.auto_balance_backing and backing_notes:
        if melody_source_insts:
            mel_med = _median_velocity_insts(melody_source_insts, default=100)
        else:
            mel_med = _median_velocity_notes(melody_notes, default=100)

        back_med = _median_velocity_notes(backing_notes, default=80)

        target = int(max(60, min(127, round(mel_med * float(config.backing_target_ratio)))))
        auto_boost = float(target) / float(max(1, back_med))
        auto_boost = max(1.0, min(float(config.max_backing_boost), auto_boost))

    bass_scale = auto_boost * float(config.backing_vel_scale) * float(config.bass_vel_scale)
    pad_scale = auto_boost * float(config.backing_vel_scale) * float(config.pad_vel_scale)
    drums_scale = auto_boost * float(config.backing_vel_scale) * float(config.drums_vel_scale)

    # Backing tracks (now with vel_scale applied)
    if bass_notes:
        pm.instruments.append(
            _notes_to_instrument(
                bass_notes,
                program=config.bass_program,
                name=DEFAULT_NAMES["bass"],
                is_drum=False,
                vel_scale=bass_scale,
            )
        )

    if pad_notes:
        pm.instruments.append(
            _notes_to_instrument(
                pad_notes,
                program=config.pad_program,
                name=DEFAULT_NAMES["pad"],
                is_drum=False,
                vel_scale=pad_scale,
            )
        )

    if drum_notes:
        pm.instruments.append(
            _notes_to_instrument(
                drum_notes,
                program=0,
                name=DEFAULT_NAMES["drums"],
                is_drum=True,
                vel_scale=drums_scale,
            )
        )

    return pm


def write_midi(
    output_path: str | Path,
    melody_notes: Sequence[Note],
    arrangement: Arrangement,
    info: MidiInfo,
    config: RenderConfig = RenderConfig(),
    melody_source_insts: Optional[Sequence[pretty_midi.Instrument]] = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pm = build_pretty_midi(
        melody_notes,
        arrangement,
        info,
        config=config,
        melody_source_insts=melody_source_insts,
    )
    pm.write(str(out))
    return out
