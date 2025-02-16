from heartbeat.beat_construction.beat_constructor import Note, LayerConfig, BPM_Manager, BeatConstructor, TimeSignature
import pandas as pd
from utils import list_files_in_directory

# Song Characteristics #
time_signature = TimeSignature(4, 4)
base_unit = Note.SIXTEENTH
########################

# LOAD DRUM PATTERNS #
drum_patterns = pd.read_csv("drum_patterns.csv").to_dict(orient='records')
drum_patterns = {item['label']: eval(item['pattern']) for item in drum_patterns}
######################

# LOAD DRUM SAMPLES #
directory_path = '../../data/drums'
drum_tracks = list_files_in_directory(directory_path)
#####################

# Generate Metronome #
bpm = 120
metronome_fn = '00.wav'
metronome_sample = None#WavSample(Note.QUARTER, "Metronome", metronome_fn)
bpm_manager = BPM_Manager(bpm, metronome_sample=metronome_sample, beat_note=Note.QUARTER, base_unit=base_unit)
######################

# Generate Beat #
layer_config0 = LayerConfig("drums", drum_tracks, drum_patterns)
layer_configs = [layer_config0]
beat = BeatConstructor.build_beat(layer_configs, time_signature, 1, base_unit)
bpm_manager.add_child(beat)
#################

# Start Beat #
bpm_manager.start()
##############

# bar0_0 = BeatBar(time_signature, base_unit)
# test_smpl0 = WavSample(Note.QUARTER, "Test0", sample_fn)
# print(pattern)
# bar0_0.load_pattern(pattern, test_smpl0)
# #bar0_0.set_count_sample(0, test_smpl0)
# #bar0_0.set_count_sample(5, test_smpl0)
# #bar0_0.set_count_sample(6, test_smpl0)
# #bar0_0.set_count_sample(7, test_smpl0)

# bar1_0 = BeatBar(time_signature, base_unit)
# test_smpl1 = WavSample(Note.QUARTER, "Test1", '02.wav')
# #bar1_0.set_count_sample(2, test_smpl1)
# #bar1_0.set_count_sample(4, test_smpl1)

# bars0 = [bar0_0]
# bars1 = [bar1_0]
# layer0 = BeatLayer(bars0)
# layer1 = BeatLayer(bars1)
# layers = [layer0, layer1]
# beat = Beat(layers)

# bpm = 120
# metronome_fn = '00.wav'
# metronome_sample = None#WavSample(Note.QUARTER, "Metronome", metronome_fn)
# bpm_manager = BPM_Manager(bpm, metronome_sample=metronome_sample, beat_note=Note.QUARTER, base_unit=base_unit)

# bpm_manager.add_child(beat)
# bpm_manager.start()