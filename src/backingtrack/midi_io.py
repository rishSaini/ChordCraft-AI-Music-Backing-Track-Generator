# src/backingtrack/midi_io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pretty_midi

from .types import BarGrid, MidiInfo, TimeSignature


DEFAULT_TEMPO_BPM = 120.0
DEFAULT_TIME_SIGNATURE = TimeSignature(4, 4)


@dataclass(frozen=True)
class MelodySelection:
    """
    What we chose as the melody track(s) from the MIDI.

    Backward compatible:
      - instrument_index / instrument_name / is_drum describe the PRIMARY melody track.
    New:
      - instrument_indices / instrument_names describe the full auto-selected melody group
        (e.g., piccolo + steel drum).
    """
    instrument_index: int
    instrument_name: str
    is_drum: bool

    instrument_indices: tuple[int, ...] = ()
    instrument_names: tuple[str, ...] = ()


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {p}")
    if p.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Expected a .mid or .midi file, got: {p.name}")

    return pretty_midi.PrettyMIDI(str(p))


def extract_midi_info(pm: pretty_midi.PrettyMIDI) -> MidiInfo:
    duration = float(pm.get_end_time())

    tempo_times, tempi = pm.get_tempo_changes()
    if len(tempi) > 0:
        tempo_bpm = float(tempi[0])
    else:
        tempo_bpm = DEFAULT_TEMPO_BPM

    if pm.time_signature_changes:
        ts0 = min(pm.time_signature_changes, key=lambda ts: ts.time)
        time_signature = TimeSignature(int(ts0.numerator), int(ts0.denominator))
    else:
        time_signature = DEFAULT_TIME_SIGNATURE

    return MidiInfo(duration=duration, tempo_bpm=tempo_bpm, time_signature=time_signature)


def build_bar_grid(info: MidiInfo, start_time: float = 0.0) -> BarGrid:
    return BarGrid(
        tempo_bpm=float(info.tempo_bpm),
        time_signature=info.time_signature,
        start_time=float(start_time),
    )


def _span(inst: pretty_midi.Instrument) -> tuple[float, float, float]:
    """(start, end, span_seconds) based on note times."""
    s = float(min(n.start for n in inst.notes))
    e = float(max(n.end for n in inst.notes))
    return s, e, max(1e-6, e - s)


def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap / min(spanA, spanB) in [0..1]."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    span_a = max(1e-6, a1 - a0)
    span_b = max(1e-6, b1 - b0)
    return float(inter / max(1e-6, min(span_a, span_b)))


def _polyphony_ratio(inst: pretty_midi.Instrument) -> float:
    """
    Approx fraction of active time where >=2 notes are sounding.
    Melody tends to be ~0, pads/chords higher.
    """
    if not inst.notes:
        return 1.0

    events: list[tuple[float, int]] = []
    for n in inst.notes:
        events.append((float(n.start), +1))
        events.append((float(n.end), -1))
    events.sort(key=lambda x: x[0])

    active = 0
    poly_time = 0.0
    total = 0.0
    for (t, d), (t2, _) in zip(events, events[1:]):
        active += d
        dt = max(0.0, t2 - t)
        total += dt
        if active >= 2:
            poly_time += dt

    return float(poly_time / max(1e-6, total))


def _chordiness(inst: pretty_midi.Instrument, window_s: float = 0.05) -> float:
    """
    Fraction of note onsets that occur "in a chord" (another onset within window_s).
    Chordal accompaniment tends to have higher chordiness.
    """
    if len(inst.notes) < 3:
        return 0.0
    starts = np.array(sorted(float(n.start) for n in inst.notes), dtype=np.float64)
    # Count onsets that have a neighbor within window_s
    diffs = np.diff(starts)
    near_next = diffs <= float(window_s)
    # onset i is chordy if near next OR previous is near_next
    chordy = np.zeros_like(starts, dtype=bool)
    chordy[:-1] |= near_next
    chordy[1:] |= near_next
    return float(chordy.mean())


def pick_melody_instrument(
    pm: pretty_midi.PrettyMIDI,
    instrument_index: Optional[int] = None,
) -> Tuple[pretty_midi.Instrument, MelodySelection]:
    """
    Choose which instrument track(s) are "the melody".

    If instrument_index is provided: pick that exact instrument (single).
    Otherwise:
      1) score tracks and pick PRIMARY melody
      2) also pick a small group of other lead-like tracks that overlap in time
         (e.g., steel drum doubling the melody)

    Return:
      - melody_inst: the PRIMARY instrument (backward-compatible)
      - selection: includes .instrument_indices for the whole melody group
    """
    instruments = pm.instruments
    if not instruments:
        raise ValueError("No instruments found in MIDI.")

    # Explicit index override (single-track)
    if instrument_index is not None:
        if instrument_index < 0 or instrument_index >= len(instruments):
            raise IndexError(
                f"instrument_index out of range: {instrument_index} "
                f"(valid 0..{len(instruments)-1})"
            )
        inst = instruments[instrument_index]
        sel = MelodySelection(
            instrument_index=instrument_index,
            instrument_name=inst.name or f"Instrument {instrument_index}",
            is_drum=bool(inst.is_drum),
            instrument_indices=(instrument_index,),
            instrument_names=((inst.name or f"Instrument {instrument_index}"),),
        )
        return inst, sel

    # Score candidates
    song_end = float(pm.get_end_time()) or 1.0

    best_idx: Optional[int] = None
    best_score = -1e18

    feats: dict[int, dict] = {}

    for idx, inst in enumerate(instruments):
        if inst.is_drum or not inst.notes:
            continue

        pitches = np.array([n.pitch for n in inst.notes], dtype=np.float32)
        note_count = len(inst.notes)

        s, e, span = _span(inst)
        coverage = span / max(1e-6, song_end)
        end_ratio = e / max(1e-6, song_end)

        median_pitch = float(np.median(pitches))
        p90_pitch = float(np.percentile(pitches, 90))

        poly = _polyphony_ratio(inst)
        chordy = _chordiness(inst, window_s=0.05)

        onset_rate = float(note_count / max(1e-6, span))  # notes/sec

        name = (inst.name or "").strip().lower()
        penalty = 0.0
        bonus = 0.0

        for bad in ("bass", "pad", "chord", "harmony", "accomp", "comp", "rhythm"):
            if bad in name:
                penalty += 25.0

        for good in ("melody", "lead", "vocal", "voice", "solo", "theme"):
            if good in name:
                bonus += 30.0

        # discourage tiny one-off/FX tracks
        short_pen = 0.0
        if coverage < 0.18:
            short_pen = 140.0 * (0.18 - coverage) / 0.18

        # Scoring:
        # - still likes higher pitch, but less dominant than before
        # - heavily rewards coverage/end
        # - penalizes polyphony/chordiness (pads)
        score = (
            0.75 * median_pitch
            + 0.45 * p90_pitch
            + 6.0 * np.log1p(note_count)
            + 135.0 * coverage
            + 30.0 * end_ratio
            + 10.0 * np.log1p(onset_rate)
            + bonus
            - penalty
            - 90.0 * poly
            - 70.0 * chordy
            - short_pen
        )

        feats[idx] = {
            "score": float(score),
            "start": s,
            "end": e,
            "span": span,
            "coverage": float(coverage),
            "end_ratio": float(end_ratio),
            "median_pitch": float(median_pitch),
            "p90_pitch": float(p90_pitch),
            "poly": float(poly),
            "chordy": float(chordy),
            "note_count": int(note_count),
        }

        if score > best_score:
            best_score = float(score)
            best_idx = int(idx)

    # Fallback: first with notes
    if best_idx is None:
        for idx, inst in enumerate(instruments):
            if inst.notes:
                best_idx = idx
                break
    if best_idx is None:
        raise ValueError("MIDI contains no notes in any instrument track.")

    # Pick a melody GROUP: other lead-like tracks near the top score that overlap in time
    max_tracks = 4
    score_margin = 35.0      # include tracks within best_score - margin
    min_overlap = 0.30       # must overlap primary in time
    max_poly = 0.35          # avoid chordal/pad tracks
    max_chordy = 0.40

    primary = feats[best_idx]
    picked = [best_idx]

    # sort by score descending
    candidates = sorted(
        (i for i in feats.keys() if i != best_idx),
        key=lambda i: feats[i]["score"],
        reverse=True,
    )

    for i in candidates:
        if len(picked) >= max_tracks:
            break

        f = feats[i]
        if f["score"] < (best_score - score_margin):
            break  # since sorted, remaining will be worse

        if f["poly"] > max_poly or f["chordy"] > max_chordy:
            continue

        ov = _overlap_ratio(primary["start"], primary["end"], f["start"], f["end"])
        if ov < min_overlap:
            continue

        # keep if it looks "melody-ish" (not way down in pitch)
        # allow some distance (steel drum etc.), but reject very low comp tracks
        if f["p90_pitch"] < (primary["median_pitch"] - 18.0):
            continue

        picked.append(i)

    picked = tuple(int(i) for i in picked)
    names = tuple((instruments[i].name or f"Instrument {i}") for i in picked)

    inst = instruments[best_idx]
    sel = MelodySelection(
        instrument_index=int(best_idx),
        instrument_name=inst.name or f"Instrument {best_idx}",
        is_drum=bool(inst.is_drum),
        instrument_indices=picked,
        instrument_names=names,
    )
    return inst, sel


def load_and_prepare(
    path: str | Path,
    melody_instrument_index: Optional[int] = None,
) -> Tuple[pretty_midi.PrettyMIDI, MidiInfo, BarGrid, pretty_midi.Instrument, MelodySelection]:
    pm = load_midi(path)
    info = extract_midi_info(pm)
    grid = build_bar_grid(info)

    melody_inst, selection = pick_melody_instrument(pm, instrument_index=melody_instrument_index)
    return pm, info, grid, melody_inst, selection
