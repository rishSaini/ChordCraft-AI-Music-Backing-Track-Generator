# src/backingtrack/cli.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .midi_io import load_and_prepare
from .melody import MelodyConfig, extract_melody_notes
from .key_detect import estimate_key, key_to_string
from .moods import apply_mood_to_key, get_mood, list_moods
from .harmony_baseline import generate_chords
from .arrange import arrange_backing
from .render import RenderConfig, write_midi


app = typer.Typer(add_completion=False, help="AI MIDI backing-track generator (v1 baseline).")


@app.command()
def generate(
    input_midi: Path = typer.Argument(..., exists=True, readable=True, help="Input MIDI file (.mid/.midi)"),
    output_midi: Path = typer.Option(Path("data/generated/out.mid"), "--out", "-o", help="Output MIDI path"),
    mood: str = typer.Option("neutral", "--mood", "-m", help=f"Mood preset. Options: {', '.join(list_moods())}"),
    melody_track: Optional[int] = typer.Option(None, "--melody-track", help="Instrument index to treat as melody (0-based)"),
    bars_per_chord: int = typer.Option(1, "--bars-per-chord", help="How many bars each chord lasts"),
    quantize_melody: bool = typer.Option(False, "--quantize-melody", help="Quantize melody note times to the beat grid"),
    no_drums: bool = typer.Option(False, "--no-drums", help="Disable drum track generation"),
    no_bass: bool = typer.Option(False, "--no-bass", help="Disable bass track generation"),
    no_pad: bool = typer.Option(False, "--no-pad", help="Disable pad track generation"),
):
    """
    Generate a backing track (bass/pad/drums) for a MIDI melody.
    """
    # 1) Load MIDI, extract global info + grid, pick melody instrument
    pm, info, grid, melody_inst, sel = load_and_prepare(input_midi, melody_instrument_index=melody_track)

    typer.echo(f"Picked melody track: idx={sel.instrument_index}, name='{sel.instrument_name}', is_drum={sel.is_drum}")
    typer.echo(f"Tempo: {info.tempo_bpm:.2f} BPM | Time signature: {info.time_signature.numerator}/{info.time_signature.denominator}")
    typer.echo(f"Duration: {info.duration:.2f}s")

    # 2) Extract melody notes (monophonic)
    mel_cfg = MelodyConfig(quantize_to_beat=quantize_melody)
    melody_notes = extract_melody_notes(melody_inst, grid=grid, config=mel_cfg)

    if not melody_notes:
        raise typer.Exit(code=1)

    typer.echo(f"Melody notes extracted: {len(melody_notes)}")

    # 3) Detect key, apply mood “nudge”
    mood_preset = get_mood(mood)
    raw_key = estimate_key(melody_notes)
    key = apply_mood_to_key(raw_key, mood_preset)

    typer.echo(f"Detected key: {key_to_string(raw_key)}")
    if key != raw_key:
        typer.echo(f"After mood '{mood_preset.name}' bias: {key_to_string(key)}")

    # 4) Generate chords
    chords = generate_chords(
        key=key,
        grid=grid,
        duration_seconds=info.duration,
        mood=mood_preset,
        melody_notes=melody_notes,
        bars_per_chord=bars_per_chord,
    )
    typer.echo(f"Chords generated: {len(chords)} (bars_per_chord={bars_per_chord})")

    # 5) Arrange backing tracks
    arrangement = arrange_backing(
        chords=chords,
        grid=grid,
        mood=mood_preset,
        make_bass=not no_bass,
        make_pad=not no_pad,
        make_drums=not no_drums,
    )

    counts = {k: len(v) for k, v in arrangement.tracks.items()}
    typer.echo(f"Backing note counts: {counts}")

    # 6) Render + write MIDI
    render_cfg = RenderConfig(write_metadata=True)
    out_path = write_midi(output_midi, melody_notes, arrangement, info, config=render_cfg)
    typer.echo(f"Wrote: {out_path.resolve()}")


def main():
    app()


if __name__ == "__main__":
    main()
