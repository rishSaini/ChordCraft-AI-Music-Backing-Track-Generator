# src/backingtrack/cli.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pretty_midi
import typer

from .midi_io import load_and_prepare
from .melody import MelodyConfig, extract_melody_notes
from .key_detect import estimate_key, key_to_string
from .moods import apply_mood_to_key, get_mood, list_moods
from .harmony_baseline import generate_chords
from .arrange import arrange_backing
from .render import RenderConfig, write_midi
from .humanize import HumanizeConfig, humanize_arrangement


app = typer.Typer(add_completion=False, help="AI MIDI backing-track generator (v1 baseline).")

def _parse_indices(csv: Optional[str]) -> list[int]:
    if not csv:
        return []
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))

@app.command()
def generate(
    input_midi: Path = typer.Argument(..., exists=True, readable=True, help="Input MIDI file (.mid/.midi)"),
    output_midi: Path = typer.Option(Path("data/generated/out.mid"), "--out", "-o", help="Output MIDI path"),
    mood: str = typer.Option("neutral", "--mood", "-m", help=f"Mood preset. Options: {', '.join(list_moods())}"),
    melody_tracks: Optional[str] = typer.Option(
        None,
        "--melody-tracks",
        help="Comma-separated instrument indices to render as the lead (e.g. 0,2,5). If omitted, auto-pick is used.",
    ),
    bars_per_chord: int = typer.Option(1, "--bars-per-chord", help="How many bars each chord lasts"),
    quantize_melody: bool = typer.Option(False, "--quantize-melody", help="Quantize melody note times to the beat grid"),
    no_drums: bool = typer.Option(False, "--no-drums", help="Disable drum track generation"),
    no_bass: bool = typer.Option(False, "--no-bass", help="Disable bass track generation"),
    no_pad: bool = typer.Option(False, "--no-pad", help="Disable pad track generation"),
    humanize: bool = typer.Option(True, "--humanize/--no-humanize", help="Apply humanization (timing/velocity/swing)"),
    jitter_ms: float = typer.Option(15.0, "--jitter-ms", help="Timing jitter amount in milliseconds"),
    vel_jitter: int = typer.Option(8, "--vel-jitter", help="Velocity jitter amount"),
    swing: float = typer.Option(0.15, "--swing", help="Swing amount 0..1 (offbeat delay)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed (also used for arrangement)"),
    structure: str = typer.Option("none", "--structure", help="Song structure: none | auto"),
    drums_mode: str = typer.Option("rules", "--drums-mode", help="Drums: rules | ml"),
    ml_temp: float = typer.Option(1.05, "--ml-temp", help="ML drum temperature (if drums_mode=ml)"),
):
    # Load + auto-pick melody for baseline metadata/grid
    requested = _parse_indices(melody_tracks)
    first_idx = requested[0] if requested else None

    pm, info, grid, melody_inst, sel = load_and_prepare(input_midi, melody_instrument_index=first_idx)

    # Decide which instruments are the "lead"
    if requested:
        for i in requested:
            if i < 0 or i >= len(pm.instruments):
                raise typer.BadParameter(f"melody track index {i} out of range (0..{len(pm.instruments)-1})")
        melody_source_insts = [pm.instruments[i] for i in requested]
        typer.echo(f"Using melody tracks (lead): {requested}")
    else:
        melody_source_insts = [melody_inst]
        typer.echo(f"Auto-picked melody track: idx={sel.instrument_index}, name='{sel.instrument_name}', is_drum={sel.is_drum}")

    typer.echo(f"Tempo: {info.tempo_bpm:.2f} BPM | Time signature: {info.time_signature.numerator}/{info.time_signature.denominator}")
    typer.echo(f"Duration: {info.duration:.2f}s")

    # Build one combined instrument for analysis (key/chords) across all selected lead tracks
    analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
    analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
    analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

    mel_cfg = MelodyConfig(quantize_to_beat=quantize_melody)
    melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=mel_cfg)
    if not melody_notes:
        raise typer.Exit(code=1)

    typer.echo(f"Melody notes extracted: {len(melody_notes)}")

    mood_preset = get_mood(mood)
    raw_key = estimate_key(melody_notes)
    key = apply_mood_to_key(raw_key, mood_preset)

    typer.echo(f"Detected key: {key_to_string(raw_key)}")
    if key != raw_key:
        typer.echo(f"After mood '{mood_preset.name}' bias: {key_to_string(key)}")

    chords = generate_chords(
        key=key,
        grid=grid,
        duration_seconds=info.duration,
        mood=mood_preset,
        melody_notes=melody_notes,
        bars_per_chord=bars_per_chord,
    )
    typer.echo(f"Chords generated: {len(chords)} (bars_per_chord={bars_per_chord})")

    arrangement = arrange_backing(
        chords=chords,
        grid=grid,
        mood=mood_preset,
        make_bass=not no_bass,
        make_pad=not no_pad,
        make_drums=not no_drums,
        seed=seed,
        structure_mode=structure,
        drums_mode=drums_mode,
        ml_drums_model_path="data/ml/drum_model.pt",
        ml_drums_temperature=ml_temp,
    )

    if humanize:
        arrangement = humanize_arrangement(
            arrangement,
            grid,
            HumanizeConfig(timing_jitter_ms=jitter_ms, velocity_jitter=vel_jitter, swing=swing, seed=seed),
        )

    counts = {k: len(v) for k, v in arrangement.tracks.items()}
    typer.echo(f"Backing note counts: {counts}")

    # Render: pass melody_source_insts so we preserve CC64 sustain/etc across ALL lead tracks
    render_cfg = RenderConfig(melody_program=int(melody_source_insts[0].program))
    write_midi(output_midi, [], arrangement, info, config=render_cfg, melody_source_insts=melody_source_insts)
    typer.echo(f"Wrote: {output_midi.resolve()}")
    

def main():
    app()


if __name__ == "__main__":
    main()
