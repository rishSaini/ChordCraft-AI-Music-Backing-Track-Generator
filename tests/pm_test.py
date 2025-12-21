import pretty_midi

pm = pretty_midi.PrettyMIDI("C:/Users/risha/Downloads/backing_track (4).mid")
for i, inst in enumerate(pm.instruments):
    print(i, inst.name, "program=", inst.program, "is_drum=", inst.is_drum, "notes=", len(inst.notes))
