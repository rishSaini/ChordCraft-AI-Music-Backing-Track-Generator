# src/backingtrack/arrange.py

from __future__ import annotations
import random

from .structure import SongStructure, auto_structure
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .types import BarGrid, Note, TimeSignature
from .moods import MoodPreset
from .harmony_baseline import ChordEvent


TrackName = Literal["bass", "pad", "drums"]


@dataclass(frozen=True)
class Arrangement:
    """
    Container for generated backing-track notes.
    Each track is a list[Note] in absolute seconds.
    """
    tracks: Dict[TrackName, List[Note]]


# --- GM drum note numbers (General MIDI percussion) ---
KICK = 36
SNARE = 38
CLOSED_HH = 42
OPEN_HH = 46
CRASH = 49
LOW_TOM = 45
MID_TOM = 47
HIGH_TOM = 50
RIDE = 51

def arrange_backing(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    make_bass: bool = True,
    make_pad: bool = True,
    make_drums: bool = True,
    bass_octave: int = 2,
    pad_octave: int = 4,
    seed: Optional[int] = None,
    structure_mode: str = "none",   # "none" or "auto"
    drums_mode: str = "rules",      # "rules" or "ml"
    ml_drums_model_path: str = "data/ml/drum_model.pt",
    ml_drums_temperature: float = 1.05,
) -> Arrangement:
    rng = random.Random(seed)

    # total bars from chords
    total_end = max((c.end for c in chords), default=0.0)
    num_bars = grid.bar_index_at(max(grid.start_time, total_end - 1e-6)) + 1

    structure: Optional[SongStructure] = None
    if structure_mode == "auto":
        structure = auto_structure(
            num_bars=num_bars,
            base_density=float(mood.rhythm_density),
            base_brightness=float(mood.brightness),
        )

    tracks: Dict[TrackName, List[Note]] = {"bass": [], "pad": [], "drums": []}

    if make_bass:
        tracks["bass"] = _make_bass(chords, grid, mood, bass_octave=bass_octave, rng=rng, structure=structure)

    if make_pad:
        tracks["pad"] = _make_pad(chords, grid, mood, pad_octave=pad_octave, structure=structure)

    if make_drums:
        if drums_mode == "ml":
            try:
                tracks["drums"] = _make_drums_ml(
                    chords,
                    grid,
                    mood,
                    seed=seed,
                    num_bars=num_bars,
                    structure=structure,
                    model_path=ml_drums_model_path,
                    temperature=ml_drums_temperature,
                )
            except ValueError:
                tracks["drums"] = _make_drums(chords, grid, mood, rng=rng, structure=structure)
        else:
            tracks["drums"] = _make_drums(chords, grid, mood, rng=rng, structure=structure)

    return Arrangement(tracks=tracks)

# -------------------------
# Bass
# -------------------------

def _make_bass(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    bass_octave: int,
    rng: random.Random,
    structure: Optional[SongStructure] = None,
) -> List[Note]:
    out: List[Note] = []

    base_pitch = 12 * bass_octave
    center = base_pitch + 16
    spb = float(grid.seconds_per_beat)

    for i, ch in enumerate(chords):
        next_ch = chords[i + 1] if i + 1 < len(chords) else None
        root = _nearest_midi_for_pc(ch.root_pc, around=center, lo=28, hi=55)

        start_bar = grid.bar_index_at(ch.start)
        end_bar = grid.bar_index_at(max(ch.start, ch.end - 1e-6))

        for bar in range(start_bar, end_bar + 1):
            style = structure.style_for_bar(bar) if structure else None
            density = float(style.density) if style else float(mood.rhythm_density)
            inten = float(style.intensity) if style else 0.6

            bar_start = grid.time_at(bar, 0.0)
            bar_end = grid.time_at(bar + 1, 0.0)
            seg_start = max(ch.start, bar_start)
            seg_end = min(ch.end, bar_end)
            if seg_end <= seg_start:
                continue

            beats = int(grid.time_signature.numerator)

            phrase2 = (bar // 2) % 2
            variant = "A" if phrase2 == 0 else "B"
            if density >= 0.8 and rng.random() < 0.25:
                variant = "B" if variant == "A" else "A"

            hits: list[tuple[float, str]] = []
            if density <= 0.40:
                hits = [(0.0, "root")]
            elif density <= 0.70:
                hits = [(0.0, "root"), (1.5, "fifth"), (2.0, "root")] if variant == "A" \
                    else [(0.0, "root"), (1.0, "oct"), (2.0, "root"), (2.5, "fifth")]
            else:
                hits = [(0.0, "root"), (0.5, "fifth"), (1.0, "root"), (1.5, "oct"),
                        (2.0, "root"), (2.5, "fifth"), (3.0, "root")] if variant == "A" \
                    else [(0.0, "root"), (0.5, "oct"), (1.0, "root"), (1.5, "fifth"),
                          (2.0, "root"), (2.5, "oct"), (3.0, "root"), (3.5, "fifth")]

            hits = [(b, k) for (b, k) in hits if b < beats]

            for beat, kind in hits:
                t0 = bar_start + beat * spb
                if not (seg_start <= t0 < seg_end):
                    continue

                dur = (0.42 if density > 0.4 else 0.55) * spb
                t1 = min(seg_end, t0 + dur)

                pitch = root
                if kind == "fifth":
                    pitch = _nearest_midi_for_pc((ch.root_pc + 7) % 12, around=root + 7, lo=28, hi=60)
                elif kind == "oct":
                    pitch = min(60, root + 12)

                base_vel = 70 + int(35 * inten)
                vel = base_vel + (6 if beat in (0.0, 2.0) else -4)
                vel = int(max(1, min(127, vel)))

                out.append(Note(pitch=int(pitch), start=float(t0), end=float(t1), velocity=vel))

            if next_ch is not None and beats >= 4 and density >= 0.45:
                approach_t = bar_start + 3.5 * spb
                if seg_start <= approach_t < seg_end:
                    prefer_pcs = set(_ordered_chord_pcs(ch)) | set(_ordered_chord_pcs(next_ch))
                    approach_pitch = _choose_approach_pitch(
                        current_root_pitch=root,
                        next_root_pc=int(next_ch.root_pc) % 12,
                        prefer_pcs=prefer_pcs,
                        lo=28,
                        hi=60,
                    )
                    out.append(Note(
                        pitch=int(approach_pitch),
                        start=float(approach_t),
                        end=float(min(seg_end, approach_t + 0.30 * spb)),
                        velocity=int(max(1, min(127, 58 + int(28 * inten)))),
                    ))

    return out

def _choose_approach_pitch(
    *,
    current_root_pitch: int,
    next_root_pc: int,
    prefer_pcs: set[int],
    lo: int,
    hi: int,
) -> int:
    """
    Pick a short approach note leading into next_root_pc.
    We prefer a 1–2 semitone step that is in either current/next chord tones,
    otherwise fall back to closest chromatic step.
    """
    # Candidate pitch-classes to approach the next root
    candidates_pc = [
        (next_root_pc - 2) % 12,
        (next_root_pc - 1) % 12,
        (next_root_pc + 1) % 12,
        (next_root_pc + 2) % 12,
    ]

    # Prefer candidates that appear in chord tones (sounds more “in-style”)
    preferred = [pc for pc in candidates_pc if pc in prefer_pcs]
    if preferred:
        candidates_pc = preferred

    # Choose the candidate closest to current_root_pitch
    best_pitch = None
    best_dist = 1e9
    for pc in candidates_pc:
        p = _nearest_midi_for_pc(pc, around=current_root_pitch, lo=lo, hi=hi)
        d = abs(p - current_root_pitch)
        if d < best_dist:
            best_dist = d
            best_pitch = p

    return int(best_pitch if best_pitch is not None else current_root_pitch)


# -------------------------
# Pad
# -------------------------

_TRIAD_INTERVALS: dict[str, Tuple[int, int, int]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}

def _make_pad(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    pad_octave: int,
    structure: Optional[SongStructure] = None,
) -> List[Note]:
    out: List[Note] = []

    max_notes = 3 if mood.rhythm_density < 0.55 else 4
    prev_voicing: Optional[List[int]] = None

    for ch in chords:
        bar = grid.bar_index_at(ch.start)
        style = structure.style_for_bar(bar) if structure else None

        # brightness comes from structure if enabled; otherwise mood.brightness
        bright = float(style.brightness) if style else float(mood.brightness)
        shift = int(round(bright * 8))
        target = 12 * pad_octave + 12 + shift

        pcs = _ordered_chord_pcs(ch)[:max_notes]
        candidates = _voicing_candidates(pcs, around=target, lo=48, hi=92)
        if not candidates:
            continue

        if prev_voicing is None:
            voicing = candidates[0]
        else:
            voicing = min(candidates, key=lambda v: _voicing_cost(v, prev_voicing))
        prev_voicing = voicing

        inten = float(style.intensity) if style else 0.6
        vel = int(max(40, min(85, 50 + 35 * inten)))

        start = ch.start
        end = max(start + 0.08, ch.end - 0.01)
        for p in voicing:
            out.append(Note(pitch=p, start=start, end=end, velocity=vel))

    return out


def _ordered_chord_pcs(ch: ChordEvent) -> List[int]:
    """
    Return chord pitch-classes in a musically sensible order:
    root, third/second/fourth, fifth, then extensions.
    """
    if ch.quality not in _TRIAD_INTERVALS:
        # fallback: just use whatever pitch_classes() gives
        return list(ch.pitch_classes())

    triad = _TRIAD_INTERVALS[ch.quality]
    pcs = [ (ch.root_pc + iv) % 12 for iv in triad ]
    pcs += [ (ch.root_pc + iv) % 12 for iv in ch.extensions ]

    # dedup, preserve order
    out: List[int] = []
    for pc in pcs:
        if pc not in out:
            out.append(pc)
    return out


def _voicing_from_pcs(pcs: Sequence[int], *, around: int, max_notes: int) -> List[int]:
    """
    Convert pitch-classes to MIDI pitches near 'around', creating a clean ascending voicing.
    """
    pcs = list(pcs)[:max_notes]

    # First pass: map each pc near around
    pitches = [_nearest_midi_for_pc(pc, around=around, lo=36, hi=96) for pc in pcs]
    pitches.sort()

    # Ensure strictly ascending (avoid collisions like same pitch)
    fixed: List[int] = []
    for p in pitches:
        if not fixed:
            fixed.append(p)
            continue
        while p <= fixed[-1]:
            p += 12
        if p <= 127:
            fixed.append(p)

    # Clamp and return
    return [min(127, max(0, p)) for p in fixed]


def _nearest_midi_for_pc(pc: int, *, around: int, lo: int, hi: int) -> int:
    """
    Find a MIDI pitch with pitch-class pc that is closest to 'around', clamped to [lo, hi].
    """
    pc = int(pc) % 12
    around = int(around)

    # Candidate pitches: around +/- a few octaves
    candidates: List[int] = []
    base = around - ((around % 12) - pc)
    for k in range(-5, 6):
        candidates.append(base + 12 * k)

    # Choose closest within bounds (or closest overall then clamp)
    best = min(candidates, key=lambda p: abs(p - around))
    best = max(lo, min(hi, best))
    return best


# -------------------------
# Drums
# -------------------------

def _make_drums_ml(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    seed: Optional[int],
    num_bars: int,
    structure: Optional[SongStructure],
    model_path: str,
    temperature: float,
) -> List[Note]:
    """
    ML drums generator.
    Uses your trained model (data/ml/drum_model.pt) to generate num_bars bars,
    then optionally applies section rules (intro no-drums, ride in chorus, intensity scaling).
    """
    # Lazy import so torch isn’t required unless you actually use ML drums
    from backingtrack.ml_drums.infer import SampleConfig, generate_ml_drums

    scfg = SampleConfig(
        bars=int(num_bars),
        temperature=float(temperature),
        stochastic=True,
        seed=seed,
        intensity=1.0,  # we’ll scale per-bar below using structure intensity
    )

    notes = generate_ml_drums(model_path, grid, scfg=scfg)

    if not structure:
        return notes

    out: List[Note] = []
    for n in notes:
        bar = grid.bar_index_at(n.start)
        st = structure.style_for_bar(bar)

        # Respect intro sections where drums are off
        if not st.drums_enabled:
            continue

        pitch = n.pitch

        # Very simple “hat type” enforcement:
        # if section wants ride, convert closed hats to ride
        if st.hat == "ride" and pitch == CLOSED_HH:
            pitch = RIDE
        if st.hat == "closed" and pitch == RIDE:
            pitch = CLOSED_HH

        # Scale velocity by section intensity (keeps chorus hotter than verse)
        inten = float(st.intensity)
        scale = 0.55 + 0.70 * inten  # ~0.8 in verses, ~1.1 in choruses
        vel = int(max(1, min(127, round(n.velocity * scale))))

        out.append(Note(pitch=int(pitch), start=float(n.start), end=float(n.end), velocity=int(vel)))

    return out

def _make_drums(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    rng: random.Random,
    structure: Optional[SongStructure] = None,
) -> List[Note]:
    out: List[Note] = []

    total_end = max((c.end for c in chords), default=0.0)
    if total_end <= 0:
        return out

    num_bars = grid.bar_index_at(max(grid.start_time, total_end - 1e-6)) + 1
    beats = int(grid.time_signature.numerator)
    spb = float(grid.seconds_per_beat)

    for bar in range(num_bars):
        style = structure.style_for_bar(bar) if structure else None
        density = float(style.density) if style else float(mood.rhythm_density)
        inten = float(style.intensity) if style else 0.6
        drums_on = bool(style.drums_enabled) if style else True
        hat = style.hat if style else "closed"
        fill_every = int(style.fill_every) if style else (8 if density < 0.70 else 4)

        if not drums_on:
            continue

        bar_start = float(grid.time_at(bar, 0.0))

        # Crash at phrase starts
        if bar % 8 == 0:
            out.append(Note(pitch=CRASH, start=bar_start, end=bar_start + 0.14, velocity=90 + int(20 * inten)))

        if beats != 4:
            _drums_generic_bar(out, bar_start, spb, beats, density, 0.5)
            continue

        hat_step_beats = 0.5 if density < 0.75 else 0.25
        hat_note = CLOSED_HH if hat == "closed" else (RIDE if hat == "ride" else CLOSED_HH)

        phrase2 = (bar // 2) % 2
        groove = "A" if phrase2 == 0 else "B"
        if density >= 0.85 and rng.random() < 0.20:
            groove = "B" if groove == "A" else "A"

        if groove == "A":
            kick_beats = [0.0, 1.5, 2.0] if density >= 0.55 else [0.0, 2.0]
            snare_beats = [1.0, 3.0]
            ghost_beats = [0.75, 2.75] if density >= 0.45 else []
            open_hat_beats = [3.5] if (hat != "closed" and density >= 0.70) else []
        else:
            kick_beats = [0.0, 2.5]
            if density >= 0.65:
                kick_beats += [3.0]
            if density >= 0.85:
                kick_beats += [1.0]
            snare_beats = [1.0, 3.0]
            ghost_beats = [1.25, 3.25] if density >= 0.5 else []
            open_hat_beats = [1.5] if (hat != "closed" and density >= 0.70) else []

        def v(base: int) -> int:
            return int(max(1, min(127, base + int(24 * inten))))

        for b in kick_beats:
            t0 = bar_start + b * spb
            out.append(Note(pitch=KICK, start=t0, end=t0 + 0.10, velocity=v(90)))

        for b in snare_beats:
            t0 = bar_start + b * spb
            out.append(Note(pitch=SNARE, start=t0, end=t0 + 0.10, velocity=v(78)))

        for b in ghost_beats:
            t0 = bar_start + b * spb
            out.append(Note(pitch=SNARE, start=t0, end=t0 + 0.07, velocity=v(22)))

        hb = 0.0
        while hb < beats - 1e-9:
            t0 = bar_start + hb * spb
            on_beat = abs((hb % 1.0) - 0.0) < 1e-6
            out.append(Note(pitch=hat_note, start=t0, end=t0 + 0.06, velocity=v(58 if on_beat else 44)))
            hb += hat_step_beats

        # Open hats if requested
        for b in open_hat_beats:
            t0 = bar_start + b * spb
            out.append(Note(pitch=OPEN_HH, start=t0, end=t0 + 0.12, velocity=v(52)))

        # Fills at section cadence
        if (bar + 1) % fill_every == 0 and density >= 0.45:
            which = (bar // fill_every) % 3
            if density >= 0.8 and rng.random() < 0.25:
                which = rng.choice([0, 1, 2])
            if which == 0:
                _add_tom_fill(out, bar_start, spb)
            elif which == 1:
                _add_snare_roll_fill(out, bar_start, spb)
            else:
                _add_kick_run_fill(out, bar_start, spb)

    return [n for n in out if n.start >= grid.start_time and n.end > n.start]


def _add_tom_fill(out: List[Note], bar_start: float, spb: float) -> None:
    """
    Simple 1-bar fill in the last beat (beat 4):
    16ths: high tom -> mid tom -> low tom -> snare
    """
    base = bar_start + 3.0 * spb  # start of beat 4
    step = 0.25 * spb            # 16th note

    seq = [
        (HIGH_TOM, 78),
        (MID_TOM, 80),
        (LOW_TOM, 84),
        (SNARE, 90),
    ]

    for i, (pitch, vel) in enumerate(seq):
        t0 = base + i * step
        out.append(Note(pitch=pitch, start=t0, end=t0 + 0.08, velocity=vel))

def _drums_generic_bar(out: List[Note], bar_start: float, spb: float, beats: int, density: float, hat_step_beats: float) -> None:
    # Kick on 1, snare mid-bar, hats
    out.append(Note(pitch=KICK, start=bar_start, end=bar_start + 0.10, velocity=106))
    if beats > 1:
        t0 = bar_start + (beats / 2.0) * spb
        out.append(Note(pitch=SNARE, start=t0, end=t0 + 0.10, velocity=94))

    hb = 0.0
    while hb < beats - 1e-9:
        t0 = bar_start + hb * spb
        out.append(Note(pitch=CLOSED_HH, start=t0, end=t0 + 0.06, velocity=70))
        hb += hat_step_beats


def _add_snare_roll_fill(out: List[Note], bar_start: float, spb: float) -> None:
    """
    Snare roll on beat 4: 8th -> 16ths (crescendo feel)
    """
    base = bar_start + 3.0 * spb
    # 8th notes then 16ths
    times = [0.0, 0.5, 0.75, 1.0, 1.25]  # in 16th units scaled by spb
    vels = [55, 60, 68, 78, 88]
    for dt, vel in zip(times, vels):
        t0 = base + dt * (0.25 * spb)
        out.append(Note(pitch=SNARE, start=t0, end=t0 + 0.06, velocity=vel))


def _add_kick_run_fill(out: List[Note], bar_start: float, spb: float) -> None:
    """
    Kick run on beat 4 + crash: adds energy into next phrase.
    """
    base = bar_start + 3.0 * spb
    step = 0.25 * spb  # 16ths
    for i in range(4):
        t0 = base + i * step
        out.append(Note(pitch=KICK, start=t0, end=t0 + 0.08, velocity=95))
    out.append(Note(pitch=CRASH, start=bar_start + 4.0 * spb - 0.01, end=bar_start + 4.0 * spb + 0.12, velocity=92))

def _voicing_candidates(pcs: Sequence[int], *, around: int, lo: int, hi: int) -> List[List[int]]:
    """
    Generate a few inversion-like voicings near 'around'.
    """
    pcs = list(pcs)
    if not pcs:
        return []

    candidates: List[List[int]] = []
    n = len(pcs)

    for inv in range(n):
        rot = pcs[inv:] + pcs[:inv]

        # Build ascending pitches
        p0 = _nearest_midi_for_pc(rot[0], around=around, lo=lo, hi=hi)
        pitches = [p0]

        for pc in rot[1:]:
            p = _nearest_midi_for_pc(pc, around=pitches[-1] + 7, lo=lo, hi=hi)
            while p <= pitches[-1] + 2:
                p += 12
            pitches.append(p)

        # If overall center is far from target, shift by octaves
        center = sum(pitches) / len(pitches)
        while center < around - 6 and max(pitches) + 12 <= hi:
            pitches = [p + 12 for p in pitches]
            center = sum(pitches) / len(pitches)
        while center > around + 6 and min(pitches) - 12 >= lo:
            pitches = [p - 12 for p in pitches]
            center = sum(pitches) / len(pitches)

        # Clamp and ensure sorted
        pitches = [max(lo, min(hi, p)) for p in pitches]
        pitches.sort()

        # Remove duplicates by pushing up octaves if needed
        fixed: List[int] = []
        for p in pitches:
            if not fixed:
                fixed.append(p)
                continue
            while p <= fixed[-1]:
                p += 12
            if p <= hi:
                fixed.append(p)

        if len(fixed) >= 3:
            candidates.append(fixed)

    # Dedup by tuple representation
    uniq: List[List[int]] = []
    seen = set()
    for c in candidates:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            uniq.append(c)
    return uniq


def _voicing_cost(curr: Sequence[int], prev: Sequence[int]) -> float:
    """
    Smaller cost = smoother movement.
    """
    m = min(len(curr), len(prev))
    cost = sum(abs(curr[i] - prev[i]) for i in range(m))

    # Penalize big jumps in chord "center"
    c_center = sum(curr) / len(curr)
    p_center = sum(prev) / len(prev)
    cost += 0.35 * abs(c_center - p_center)

    # Penalize extreme spread (too wide sounds thin, too tight can clash)
    spread = max(curr) - min(curr)
    if spread > 19:  # > ~a 10th
        cost += (spread - 19) * 0.25

    return cost
