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


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(path))


def extract_midi_info(pm: pretty_midi.PrettyMIDI) -> MidiInfo:
    duration = float(pm.get_end_time())

    tempo_times, tempi = pm.get_tempo_changes()  # ✅ correct order
    tempo_bpm = DEFAULT_TEMPO_BPM

    if len(tempi) > 0:
        # some MIDIs can have weird 0/negative tempo entries; pick first valid
        for t in tempi:
            t = float(t)
            if t > 0:
                tempo_bpm = t
                break

    if pm.time_signature_changes:
        ts0 = min(pm.time_signature_changes, key=lambda ts: ts.time)
        time_signature = TimeSignature(int(ts0.numerator), int(ts0.denominator))
    else:
        time_signature = DEFAULT_TIME_SIGNATURE

    return MidiInfo(duration=duration, tempo_bpm=float(tempo_bpm), time_signature=time_signature)

def build_bar_grid(info: MidiInfo, start_time: float = 0.0) -> BarGrid:
    return BarGrid(
        tempo_bpm=float(info.tempo_bpm),
        time_signature=info.time_signature,
        start_time=float(start_time),
    )


@dataclass(frozen=True)
class MelodySelection:
    instrument_index: int
    instrument_name: str
    is_drum: bool

    # NEW: group selection (multiple lead instruments)
    instrument_indices: tuple[int, ...]
    instrument_names: tuple[str, ...]


def _span(inst: pretty_midi.Instrument) -> tuple[float, float, float]:
    if not inst.notes:
        return 0.0, 0.0, 1e-6
    s = float(min(n.start for n in inst.notes))
    e = float(max(n.end for n in inst.notes))
    return s, e, max(1e-6, e - s)


def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    span_a = max(1e-6, a1 - a0)
    span_b = max(1e-6, b1 - b0)
    return float(inter / max(1e-6, min(span_a, span_b)))


def _polyphony_ratio(inst: pretty_midi.Instrument) -> float:
    """Fraction of active time where >=2 notes are sounding."""
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


def _chordiness(inst: pretty_midi.Instrument) -> float:
    """
    Rough "chordiness": fraction of notes that start within a very small window
    of another note (i.e., stacked onsets).
    """
    if len(inst.notes) < 2:
        return 0.0
    starts = sorted(float(n.start) for n in inst.notes)
    eps = 0.03
    close = 0
    for a, b in zip(starts, starts[1:]):
        if (b - a) <= eps:
            close += 1
    return float(close / max(1, len(starts) - 1))


def _is_bass_like(program: int, median_pitch: float, p90: float, low_cut: float, name: str) -> bool:
    name_l = (name or "").lower()

    # Strong GM bass program block
    if 32 <= int(program) <= 39:
        return True

    # Strong name cues (treat as bass even if played high)
    bass_tokens = ("bass", "fretless", "upright", "contrabass", "subbass")
    if any(tok in name_l for tok in bass_tokens):
        return True

    # Pitch-based backup
    if median_pitch < (low_cut + 10.0) and p90 < (low_cut + 16.0):
        return True

    return False


def pick_melody_instrument(
    pm: pretty_midi.PrettyMIDI,
    instrument_index: Optional[int] = None,
) -> tuple[pretty_midi.Instrument, MelodySelection]:
    instruments = pm.instruments

    if instrument_index is not None:
        if instrument_index < 0 or instrument_index >= len(instruments):
            raise ValueError(
                f"instrument_index {instrument_index} out of range "
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

    song_end = float(pm.get_end_time()) or 1.0

    # Build feature table
    feats: dict[int, dict] = {}
    medians: list[float] = []

    for idx, inst in enumerate(instruments):
        if inst.is_drum or not inst.notes:
            continue

        pitches = np.array([n.pitch for n in inst.notes], dtype=np.float32)
        note_count = int(len(inst.notes))

        s, e, span = _span(inst)
        coverage = float(span / max(1e-6, song_end))
        end_ratio = float(e / max(1e-6, song_end))

        median_pitch = float(np.median(pitches))
        p90 = float(np.percentile(pitches, 90))
        p10 = float(np.percentile(pitches, 10))

        poly = float(_polyphony_ratio(inst))
        chordy = float(_chordiness(inst))

        onset_rate = float(note_count / max(1e-6, span))  # notes per second (on active span)
        high = float(p90 > 96.0)  # very high register (often FX)
        low = float(p10 < 45.0)   # low register (often bass / comp)

        feats[idx] = {
            "idx": idx,
            "name": inst.name or f"Instrument {idx}",
            "program": int(inst.program),
            "notes": note_count,
            "start": float(s),
            "end": float(e),
            "span": float(span),
            "coverage": float(coverage),
            "end_ratio": float(end_ratio),
            "median": float(median_pitch),
            "p90": float(p90),
            "p10": float(p10),
            "poly": float(poly),
            "chordy": float(chordy),
            "onset_rate": float(onset_rate),
            "high": float(high),
            "low": float(low),
        }
        medians.append(median_pitch)

    if not feats:
        raise ValueError("MIDI contains no notes in non-drum instruments.")

    # dynamic low-cut based on global median
    global_median = float(np.median(np.array(medians, dtype=np.float32))) if medians else 60.0
    low_cut = max(36.0, min(55.0, global_median - 10.0))

    def melody_score(f: dict) -> float:
        # Weighted score to find a primary lead candidate (still conservative)
        pitch_term = 0.0
        if f["median"] >= low_cut:
            pitch_term += 12.0
        if f["p90"] >= (low_cut + 18.0):
            pitch_term += 12.0

        score = (
            + 140.0 * f["coverage"]
            + 30.0 * f["end_ratio"]
            + 8.0 * np.log1p(f["notes"])
            + 30.0 * np.log1p(f["onset_rate"])
            + pitch_term
            - 230.0 * f["poly"]
            - 170.0 * f["chordy"]
            - 18.0 * f["high"]
            - 25.0 * f["low"]
        )
        return float(score)

    # Filter out bass-like tracks for PRIMARY melody selection
    primary_candidates: list[int] = []
    for idx, f in feats.items():
        if _is_bass_like(f["program"], f["median"], f["p90"], low_cut, f["name"]):
            continue
        # drop tiny/FX tracks
        if f["notes"] < 20 or f["coverage"] < 0.10:
            continue
        primary_candidates.append(idx)

    # If we filtered too hard, fall back to all non-drum tracks (but still keep bass programs blocked)
    if not primary_candidates:
        for idx, f in feats.items():
            if 32 <= f["program"] <= 39:
                continue
            primary_candidates.append(idx)

    scored = sorted(primary_candidates, key=lambda i: melody_score(feats[i]), reverse=True)

    # Primary = best score
    primary_idx = scored[0]
    primary_f = feats[primary_idx]

    # Build melody GROUP: include other lead-like tracks.
    #
    # Real-world MIDI files often "spread" the lead across multiple instruments (e.g., piccolo +
    # steel drum + other melodic layers), sometimes with little/no time overlap (handoffs).
    # So the group logic should be *looser* than the primary-pick logic, while still avoiding
    # bass/pad/chordal tracks.
    score_margin = 45.0          # how far below the primary score we still accept
    max_tracks = 8
    min_overlap = 0.10           # prefer some overlap, but don't require much
    min_notes = 15
    min_coverage = 0.06          # fraction of song duration the track spans (based on start/end span)

    primary_score = melody_score(primary_f)
    group: list[int] = [primary_idx]

    for i in scored[1:]:
        if len(group) >= max_tracks:
            break
        f = feats[i]

        # keep only reasonably good melodic candidates
        if melody_score(f) < (primary_score - score_margin):
            continue

        # drop tiny/FX tracks
        if f["notes"] < min_notes or f["coverage"] < min_coverage:
            continue

        # avoid pads/chords
        if f["poly"] > 0.50 or f["chordy"] > 0.55:
            continue

        # avoid bass-like secondary too
        if _is_bass_like(f["program"], f["median"], f["p90"], low_cut, f["name"]):
            continue

        # keep generally "melodic register" (cuts out low-mid comp patterns)
        if f["p90"] < (low_cut + 18.0):
            continue

        # prefer overlap, but allow handoffs if the track is substantial
        ov = _overlap_ratio(primary_f["start"], primary_f["end"], f["start"], f["end"])
        if ov < min_overlap and f["coverage"] < 0.18:
            continue

        group.append(i)

    # If we still only found one instrument, fall back to adding the next-best lead-like tracks,
    # even if they don't overlap (common for "melody swaps" across sections).
    if len(group) == 1:
        for i in scored[1:]:
            if len(group) >= max_tracks:
                break
            f = feats[i]
            if f["notes"] < 20 or f["coverage"] < 0.08:
                continue
            if f["poly"] > 0.50 or f["chordy"] > 0.55:
                continue
            if _is_bass_like(f["program"], f["median"], f["p90"], low_cut, f["name"]):
                continue
            if f["p90"] < (low_cut + 18.0):
                continue
            group.append(i)

    # Optional: choose the "most melodic" as primary among the group
    # (prevents sparse countermelody from being primary)
    def mass_metric(f: dict) -> float:
        # favors sustained, non-chordal lead content
        return float(f["notes"] * f["coverage"] * (1.0 - f["poly"]) * (1.0 - f["chordy"]))

    primary_idx = max(group, key=lambda i: mass_metric(feats[i]))
    primary_inst = instruments[primary_idx]

    picked = tuple(int(i) for i in group)
    names = tuple((instruments[i].name or f"Instrument {i}") for i in picked)

    sel = MelodySelection(
        instrument_index=int(primary_idx),
        instrument_name=primary_inst.name or f"Instrument {primary_idx}",
        is_drum=bool(primary_inst.is_drum),
        instrument_indices=picked,
        instrument_names=names,
    )
    return primary_inst, sel


def load_and_prepare(
    path: str | Path,
    melody_instrument_index: Optional[int] = None,
) -> Tuple[pretty_midi.PrettyMIDI, MidiInfo, BarGrid, pretty_midi.Instrument, MelodySelection]:
    pm = load_midi(path)
    info = extract_midi_info(pm)
    grid = build_bar_grid(info)

    melody_inst, selection = pick_melody_instrument(pm, instrument_index=melody_instrument_index)
    return pm, info, grid, melody_inst, selection
