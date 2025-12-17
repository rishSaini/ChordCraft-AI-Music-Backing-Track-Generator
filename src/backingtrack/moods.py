# src/backingtrack/moods.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from .types import KeyEstimate, Mode

MoodName = Literal["happy", "sad", "dreamy", "tense", "neutral"]


@dataclass(frozen=True)
class MoodPreset:
    """
    A mood preset is a bundle of generation preferences.

    They’ll be used these later in:
    - harmony_baseline.py (chord templates + chord palette)
    - arrange.py (rhythm density, register, drum complexity)
    - render.py (instrument choices, velocities)

    For now, we keep it simple and practical.
    """
    name: MoodName

    # If not None, this mood prefers a mode ("major" or "minor").
    preferred_mode: Optional[Mode] = None

    # Only override the detected mode if key confidence is below this threshold.
    # Higher value = mood overrides more often.
    mode_override_if_confidence_below: float = 0.6

    # Tempo multiplier for generation
    # e.g. happy often feels a bit faster, sad often feels slower.
    tempo_multiplier: float = 1.0

    # How busy accompaniment should be (0..1). Will be used in arrange.py later.
    rhythm_density: float = 0.5

    # Suggested chord “color” / palette knobs (used later in harmony_baseline.py)
    # Triads only vs allow sevenths/add9/sus etc.
    allow_sevenths: bool = False
    allow_sus: bool = False
    allow_add9: bool = False

    # Register suggestion for accompaniment (used later).
    # -1 = darker/lower, +1 = brighter/higher
    brightness: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.mode_override_if_confidence_below <= 1.0):
            raise ValueError("mode_override_if_confidence_below must be in [0,1]")
        if self.tempo_multiplier <= 0:
            raise ValueError("tempo_multiplier must be > 0")
        if not (0.0 <= self.rhythm_density <= 1.0):
            raise ValueError("rhythm_density must be in [0,1]")
        if not (-1.0 <= self.brightness <= 1.0):
            raise ValueError("brightness must be in [-1,1]")


# ---- Preset library ----

_PRESETS: Dict[str, MoodPreset] = {
    "neutral": MoodPreset(
        name="neutral",
        preferred_mode=None,
        mode_override_if_confidence_below=0.0,  # never override
        tempo_multiplier=1.0,
        rhythm_density=0.5,
        allow_sevenths=False,
        allow_sus=False,
        allow_add9=False,
        brightness=0.0,
    ),
    "happy": MoodPreset(
        name="happy",
        preferred_mode="major",
        mode_override_if_confidence_below=0.65,
        tempo_multiplier=1.08,
        rhythm_density=0.65,
        allow_sevenths=True,   # tasteful color
        allow_sus=True,
        allow_add9=True,
        brightness=0.6,
    ),
    "sad": MoodPreset(
        name="sad",
        preferred_mode="minor",
        mode_override_if_confidence_below=0.65,
        tempo_multiplier=0.92,
        rhythm_density=0.35,
        allow_sevenths=True,   # minor7 etc. can sound expressive
        allow_sus=False,
        allow_add9=True,       # add9 can feel wistful
        brightness=-0.5,
    ),
    "dreamy": MoodPreset(
        name="dreamy",
        preferred_mode=None,   # can work in major or minor
        mode_override_if_confidence_below=0.0,  # don’t force mode
        tempo_multiplier=0.97,
        rhythm_density=0.45,
        allow_sevenths=True,
        allow_sus=True,
        allow_add9=True,
        brightness=0.2,
    ),
    "tense": MoodPreset(
        name="tense",
        preferred_mode="minor",
        mode_override_if_confidence_below=0.55,
        tempo_multiplier=1.02,
        rhythm_density=0.55,
        allow_sevenths=True,
        allow_sus=True,
        allow_add9=False,
        brightness=-0.2,
    ),
}


def get_mood(name: str) -> MoodPreset:
    """
    Fetch a mood preset by name (case-insensitive).
    Raises if unknown.
    """
    key = name.strip().lower()
    if key not in _PRESETS:
        valid = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"Unknown mood '{name}'. Valid options: {valid}")
    return _PRESETS[key]


def list_moods() -> Tuple[str, ...]:
    """Return available mood names (useful for CLI help text)."""
    return tuple(sorted(_PRESETS.keys()))


def apply_mood_to_key(key: KeyEstimate, mood: MoodPreset) -> KeyEstimate:
    """
    “Snap / nudge” the key’s mode using mood preference, but only when:
    - mood has a preferred_mode, and
    - key confidence is below the mood’s override threshold.

    We DO NOT change the tonic here — only major/minor ambiguity.
    """
    if mood.preferred_mode is None:
        return key

    if key.confidence < mood.mode_override_if_confidence_below:
        return KeyEstimate(tonic_pc=key.tonic_pc, mode=mood.preferred_mode, confidence=key.confidence)

    return key
