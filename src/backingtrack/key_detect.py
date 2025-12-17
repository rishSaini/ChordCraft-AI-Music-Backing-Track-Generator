# src/backingtrack/key_detect.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .types import KeyEstimate, Note, Mode


# Pitch class names for nice debug printing (C=0, C#=1, ..., B=11)
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# Krumhansl-Schmuckler key profiles (major/minor).
# These are "how important each pitch class tends to be" in that key.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)


@dataclass(frozen=True)
class KeyDetectConfig:
    """
    duration_weight:
      If True, weight pitch-classes by note duration (recommended).
      If False, every note contributes equally.

    use_velocity:
      If True, also weight notes by velocity (slight emphasis on "important" notes).
      Velocity weighting is mild so it doesn't dominate.

    min_total_weight:
      If melody is too short/sparse, key estimation becomes random.
      Below this threshold, we return a default guess with low confidence.
    """
    duration_weight: bool = True
    use_velocity: bool = True
    min_total_weight: float = 1.0


def _rotate_profile(profile: np.ndarray, tonic_pc: int) -> np.ndarray:
    """
    Rotate a profile so index 0 corresponds to tonic_pc.
    Example: tonic_pc=2 (D) means we compare melody histogram against "D major profile".
    """
    tonic_pc = int(tonic_pc) % 12
    return np.roll(profile, tonic_pc)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two 12-D vectors.
    Returns a value in roughly [-1, 1].
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _pitch_class_histogram(
    notes: Iterable[Note],
    *,
    duration_weight: bool,
    use_velocity: bool,
) -> np.ndarray:
    """
    Build a 12-bin histogram of pitch classes from melody notes.
    """
    hist = np.zeros(12, dtype=np.float32)

    for n in notes:
        pc = int(n.pitch) % 12

        w = 1.0
        if duration_weight:
            w *= max(0.0, float(n.end - n.start))

        if use_velocity:
            # mild velocity influence (keeps it from overpowering duration)
            v = float(getattr(n, "velocity", 100))
            w *= (0.5 + 0.5 * (v / 127.0))

        hist[pc] += w

    return hist


def estimate_key(
    melody_notes: Iterable[Note],
    config: KeyDetectConfig = KeyDetectConfig(),
) -> KeyEstimate:
    """
    Estimate (tonic_pc, mode) from a melody line.

    Returns KeyEstimate with:
      tonic_pc: 0..11
      mode: "major" or "minor"
      confidence: 0..1 (higher = more separable best-vs-runner-up)
    """
    notes = list(melody_notes)
    if not notes:
        return KeyEstimate(tonic_pc=0, mode="major", confidence=0.0)

    hist = _pitch_class_histogram(
        notes,
        duration_weight=config.duration_weight,
        use_velocity=config.use_velocity,
    )

    total = float(hist.sum())
    if total < config.min_total_weight:
        # too little info; return a safe default (C major) with low confidence
        return KeyEstimate(tonic_pc=0, mode="major", confidence=0.0)

    # Normalize histogram so it represents a distribution (scale-invariant).
    hist = hist / (hist.sum() + 1e-9)

    # Compare histogram against all 24 keys (12 major + 12 minor).
    candidates: list[Tuple[float, int, Mode]] = []

    for tonic in range(12):
        maj_prof = _rotate_profile(_MAJOR_PROFILE, tonic)
        min_prof = _rotate_profile(_MINOR_PROFILE, tonic)

        maj_score = _pearson_corr(hist, maj_prof)
        min_score = _pearson_corr(hist, min_prof)

        candidates.append((maj_score, tonic, "major"))
        candidates.append((min_score, tonic, "minor"))

    # Sort by score descending (best first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_tonic, best_mode = candidates[0]
    second_score = candidates[1][0]

    # Confidence: how much better is best than runner-up?
    # best_score is in [-1,1], so (best-second) is in [0,2].
    gap = float(best_score - second_score)

    # Map gap to 0..1 in a simple, stable way:
    # - gap ~ 0.00 -> 0.0 confidence
    # - gap ~ 0.20 -> ~0.5 confidence
    # - gap >= 0.40 -> close to 1.0
    confidence = float(np.clip(gap / 0.40, 0.0, 1.0))

    return KeyEstimate(tonic_pc=int(best_tonic), mode=best_mode, confidence=confidence)


def key_to_string(key: KeyEstimate) -> str:
    """
    Pretty-print a KeyEstimate: e.g. 'E minor (0.72)'.
    """
    tonic = PC_NAMES[int(key.tonic_pc) % 12]
    return f"{tonic} {key.mode} ({key.confidence:.2f})"
