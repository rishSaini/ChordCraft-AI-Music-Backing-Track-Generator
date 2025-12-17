# src/backingtrack/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Seconds = float
BPM = float
MidiPitch = int
Velocity = int
PitchClass = int

Mode = Literal["major", "minor"]


def _clamp_int(x: int, lo: int, hi: int, name: str) -> int:
    if x < lo or x > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {x}")
    return x


@dataclass(frozen=True)
class Note:
    pitch: MidiPitch
    start: Seconds
    end: Seconds
    velocity: Velocity = 100

    def __post_init__(self) -> None:
        _clamp_int(int(self.pitch), 0, 127, "pitch")
        _clamp_int(int(self.velocity), 0, 127, "velocity")
        if self.end <= self.start:
            raise ValueError(f"end must be > start, got start={self.start}, end={self.end}")

    @property
    def duration(self) -> Seconds:
        return self.end - self.start


@dataclass(frozen=True)
class TimeSignature:
    numerator: int = 4
    denominator: int = 4

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError(f"numerator must be > 0, got {self.numerator}")
        if self.denominator not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"denominator should be 1/2/4/8/16/32, got {self.denominator}")


@dataclass(frozen=True)
class BarGrid:
    tempo_bpm: BPM
    time_signature: TimeSignature = TimeSignature()
    start_time: Seconds = 0.0

    def __post_init__(self) -> None:
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be > 0, got {self.tempo_bpm}")

    @property
    def seconds_per_beat(self) -> Seconds:
        return 60.0 / self.tempo_bpm

    @property
    def beats_per_bar(self) -> float:
        return float(self.time_signature.numerator)

    @property
    def bar_duration(self) -> Seconds:
        return self.beats_per_bar * self.seconds_per_beat

    def quantize_time_to_beat(self, t: Seconds) -> Seconds:
        if t < self.start_time:
            return self.start_time
        beats = (t - self.start_time) / self.seconds_per_beat
        return self.start_time + round(beats) * self.seconds_per_beat


@dataclass(frozen=True)
class MidiInfo:
    duration: Seconds
    tempo_bpm: BPM = 120.0
    time_signature: TimeSignature = TimeSignature()

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError(f"duration must be >= 0, got {self.duration}")
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be > 0, got {self.tempo_bpm}")

@dataclass(frozen=True)
class KeyEstimate:
    tonic_pc: int          # 0..11
    mode: Mode             # "major" or "minor"
    confidence: float = 0.0  # 0..1

    def __post_init__(self) -> None:
        _clamp_int(int(self.tonic_pc), 0, 11, "tonic_pc")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")