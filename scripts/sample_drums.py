# scripts/sample_drums.py
from __future__ import annotations

import argparse
from pathlib import Path

import pretty_midi

from backingtrack.types import BarGrid, TimeSignature
from backingtrack.ml_drums.infer import SampleConfig, generate_ml_drums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="data/ml/drum_model.pt")
    ap.add_argument("--out", type=str, default="data/generated/ml_drums.mid")
    ap.add_argument("--bars", type=int, default=32)
    ap.add_argument("--bpm", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temp", type=float, default=1.05)
    ap.add_argument("--intensity", type=float, default=0.75)
    args = ap.parse_args()

    grid = BarGrid(tempo_bpm=float(args.bpm), time_signature=TimeSignature(4, 4), start_time=0.0)

    scfg = SampleConfig(
        bars=int(args.bars),
        temperature=float(args.temp),
        stochastic=True,
        seed=int(args.seed),
        intensity=float(args.intensity),
    )

    notes = generate_ml_drums(args.model, grid, scfg=scfg)

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(args.bpm))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, time=0.0))

    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="ML Drums")
    for n in notes:
        drum_inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end))
    drum_inst.notes.sort(key=lambda x: (x.start, x.pitch))
    pm.instruments.append(drum_inst)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
