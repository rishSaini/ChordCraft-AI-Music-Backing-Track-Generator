from backingtrack.midi_io import load_and_prepare
from backingtrack.melody import extract_melody_notes
from backingtrack.key_detect import estimate_key, key_to_string

pm, info, grid, melody_inst, sel = load_and_prepare("data/raw/85263.mid")
mel = extract_melody_notes(melody_inst, grid=grid)

k = estimate_key(mel)
print("Key:", key_to_string(k))
