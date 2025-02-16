from beat_constructor import Note, Beat, BeatLayer, BeatBar, BPM_Manager, WavSample, SampleLibrary, TimeSignature

time_signature = TimeSignature(4, 4)
base_unit = Note.EIGHTH

bar0_0 = BeatBar(time_signature, base_unit)
test_smpl0 = WavSample(Note.QUARTER, "Test0", '01.wav')
bar0_0.set_count_sample(0, test_smpl0)
bar0_0.set_count_sample(5, test_smpl0)
bar0_0.set_count_sample(6, test_smpl0)
bar0_0.set_count_sample(7, test_smpl0)

bar1_0 = BeatBar(time_signature, base_unit)
test_smpl1 = WavSample(Note.QUARTER, "Test1", '02.wav')
bar1_0.set_count_sample(2, test_smpl1)
bar1_0.set_count_sample(4, test_smpl1)

bars0 = [bar0_0]
bars1 = [bar1_0]
layer0 = BeatLayer(bars0)
layer1 = BeatLayer(bars1)
layers = [layer0, layer1]
beat = Beat(layers)

bpm = 120
metronome_fn = '00.wav'
metronome_sample = WavSample(Note.QUARTER, "Metronome", metronome_fn)
bpm_manager = BPM_Manager(bpm, metronome_sample=metronome_sample, beat_note=Note.QUARTER, base_unit=base_unit)

bpm_manager.add_child(beat)
bpm_manager.start()