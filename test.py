from beat_constructor import Note, Beat, BeatLayer, BeatBar, BPM_Manager, WavSample, SampleLibrary, TimeSignature

time_signature = TimeSignature(4, 4)
base_unit = Note.QUARTER
bar0_0 = BeatBar(time_signature, base_unit)

test_smpl = WavSample(Note.QUARTER, "Test", '01.wav')
bar0_0.set_count_sample(1, test_smpl)
bar0_0.set_count_sample(3, test_smpl)

bars0 = [bar0_0]
layer0 = BeatLayer(bars0)
layers = [layer0]
beat = Beat(layers)

bpm = 120
metronome_fn = '00.wav'
metronome_sample = WavSample(Note.QUARTER, "Metronome", metronome_fn)
bpm_manager = BPM_Manager(bpm, metronome_sample=metronome_sample, beat_note=Note.QUARTER, base_unit=base_unit)

bpm_manager.add_child(beat)
bpm_manager.start()