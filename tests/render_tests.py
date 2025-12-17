from backingtrack.midi_io import load_and_prepare
from backingtrack.melody import extract_melody_notes
from backingtrack.key_detect import estimate_key
from backingtrack.moods import get_mood, apply_mood_to_key
from backingtrack.harmony_baseline import generate_chords
from backingtrack.arrange import arrange_backing
from backingtrack.render import write_midi

pm, info, grid, melody_inst, sel = load_and_prepare("data/raw/Tokyo Ghoul - Unravel.mid")
mel = extract_melody_notes(melody_inst, grid=grid)

mood = get_mood("happy")
key = apply_mood_to_key(estimate_key(mel), mood)

chords = generate_chords(key, grid, duration_seconds=info.duration, mood=mood, melody_notes=mel)
arr = arrange_backing(chords, grid, mood)

write_midi("data/generated/out.mid", mel, arr, info)
