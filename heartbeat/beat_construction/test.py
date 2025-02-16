from heartbeat.beat_construction.beat_constructor import Note, LayerConfig, BPM_Manager, BeatConstructor, TimeSignature, WavSample
import pandas as pd
from heartbeat.beat_construction.utils import list_files_in_directory

# Song Characteristics #
time_signature = TimeSignature(4, 4)
base_unit = Note.SIXTEENTH
########################

# LOAD DRUM PATTERNS #
drum_patterns = pd.read_csv("heartbeat/beat_construction/drum_patterns.csv").to_dict(orient='records')
drum_patterns = {item['label']: eval(item['pattern']) for item in drum_patterns}
######################

# LOAD SYNTH PATTERNS #
# synth_patterns = pd.read_csv("heartbeat/beat_construction/synth_patterns.csv").to_dict(orient='records')
# synth_patterns = {item['label']: eval(item['pattern']) for item in synth_patterns}
######################

# LOAD DRUM SAMPLES #
directory_path = 'data/drums'
drum_tracks = list_files_in_directory(directory_path)
#####################

# LOAD SYNTH SAMPLES #
# directory_path = 'data/synths'
# synth_tracks = list_files_in_directory(directory_path)
#####################

# Generate Metronome #
bpm = 120
metronome_fn = '00.wav'
metronome_sample = None#WavSample(Note.QUARTER, "Metronome", metronome_fn)
bpm_manager = BPM_Manager(bpm, metronome_sample=metronome_sample, beat_note=Note.QUARTER, base_unit=base_unit)
######################

# Generate Beat #
num_bars = 2
layer_configs = []
drum_layers = 2
synth_layers = 1
for _ in range(drum_layers):
    layer_configs.append(LayerConfig("drums", drum_tracks, drum_patterns))
# for _ in range(synth_layers):
#     layer_configs.append(LayerConfig("drums", synth_tracks, synth_patterns))
beat = BeatConstructor.build_beat(layer_configs, time_signature, num_bars, base_unit)
bpm_manager.add_child(beat)
#################

# Start Beat #
bpm_manager.start()
##############