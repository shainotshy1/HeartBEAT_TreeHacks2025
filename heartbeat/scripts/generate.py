import asyncio
import pandas as pd
import time
from dataclasses import dataclass
from datetime import datetime

from heartbeat.beat_construction.beat_constructor import Note, LayerConfig, BPM_Manager, BeatConstructor, TimeSignature
from heartbeat.beat_construction.utils import list_files_in_directory
from heartbeat.heartbeat_sensor.heartbeat_sensors import HeartbeatSensor, ArduinoHeartbeatSensor, SimulatedHeartbeatSensor
from heartbeat.heartbeat_sensor.signal_processing import SignalProcessor
from heartbeat.heartbeat_sensor.emotion import Emotion


@dataclass
class MusicConfig:
    time_signature: TimeSignature
    base_unit: Note
    bpm: int
    num_bars: int
    drum_layers: int
    synth_layers: int
    emotion_run_length: float  # seconds
    sensor_buffer_size: int  # ticks
    sensor_signal_filter_chunk_len: int  # ticks
    

async def async_beat_builder(layer_configs, time_signature, num_bars, base_unit, emotion: Emotion):
    return BeatConstructor.build_beat(layer_configs, time_signature, num_bars, base_unit, emotion.value)

async def main_loop(config: MusicConfig, sensor: HeartbeatSensor, debug: bool = False, logging: bool = True):
    drum_patterns = pd.read_csv("heartbeat/beat_construction/drum_patterns.csv").to_dict(orient='records')
    drum_patterns = {item['label']: eval(item['pattern']) for item in drum_patterns}
    drum_tracks = list_files_in_directory('data/drums', '.wav')
    synth_patterns = pd.read_csv("heartbeat/beat_construction/synth_patterns.csv").to_dict(orient='records')
    synth_patterns = {item['label']: eval(item['pattern']) for item in synth_patterns}
    synth_tracks = list_files_in_directory('data/synths', '.wav')
    layer_configs = []
    for _ in range(config.drum_layers):
        layer_configs.append(LayerConfig("drums", drum_tracks, drum_patterns))
    for _ in range(config.synth_layers):
        layer_configs.append(LayerConfig("drums", synth_tracks, synth_patterns))

    signal_processor = SignalProcessor()
    
    existing_beat = None
    old_emotion = Emotion.NEUTRAL  # most recent emotion used for music generation
    existing_emotion = Emotion.NEUTRAL  # most recent computed emotion
    emotion_start_time = None
    last_logged_time = None
    should_toggle = True
    
    beat = BeatConstructor.build_beat(layer_configs, config.time_signature, config.num_bars, config.base_unit, existing_emotion)
    bpm_manager = BPM_Manager(config.bpm, metronome_sample=None, beat_note=Note.QUARTER, base_unit=config.base_unit)
    bpm_manager.add_child(beat)
    
    bpm_manager.start()

    i = 0  # sensor buffer index
    try:
        while True:
            signal_value, timestamp_str = sensor.read_signal()
            timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            if emotion_start_time is None:
                emotion_start_time = timestamp_dt
                last_logged_time = timestamp_dt
            if not debug:  # simulate heartbeat delay
                time.sleep((timestamp_dt - last_logged_time).total_seconds())
            signal_processor.update_signal(signal_value)
            
            if i > sensor.buffer_size and (i + 1) % config.sensor_signal_filter_chunk_len == 0:
                filtered_signal = signal_processor.filter_noise_ema(sensor.signal_values)
                sensor.process(filtered_signal, sensor.timestamps, timing=False)
                emotion = sensor.determine_emotion()

                # if emotion has run for long enough, change the music
                if should_toggle and (timestamp_dt - emotion_start_time).total_seconds() > config.emotion_run_length:
                    if logging:
                        print(f'==== Switching songs based on emotion: {old_emotion} -> {emotion.value}')
                    old_emotion = emotion.value
                    beat = await async_beat_builder(layer_configs, config.time_signature, config.num_bars, config.base_unit, emotion)
                    bpm_manager.remove_child(existing_beat)  # FIXME: maybe wrong
                    bpm_manager.add_child(beat)
                    existing_beat = beat
                    should_toggle = False
                    
                # mark new emotion
                if existing_emotion != emotion:
                    existing_emotion = emotion
                    emotion_start_time = timestamp_dt
                    should_toggle = True
            
            i += 1

    except KeyboardInterrupt:
        bpm_manager.stop()


def generate_synthetic(debug=False):
    config = MusicConfig(
        time_signature=TimeSignature(4, 4),
        base_unit=Note.SIXTEENTH,
        bpm=120,
        num_bars=2,
        drum_layers=3,
        synth_layers=1,
        emotion_run_length=1,
        sensor_buffer_size=1000,
        sensor_signal_filter_chunk_len=100
    )
    
    sensor = SimulatedHeartbeatSensor(buffer_size=config.sensor_buffer_size)
    asyncio.run(main_loop(config, sensor, debug))
    
    
def generate_arduino(debug=False):
    config = MusicConfig(
        time_signature=TimeSignature(4, 4),
        base_unit=Note.SIXTEENTH,
        bpm=120,
        num_bars=2,
        drum_layers=2,
        synth_layers=1,
        emotion_run_length=1,
        sensor_buffer_size=1000,
        sensor_signal_filter_chunk_len=100
    )
    
    sensor = ArduinoHeartbeatSensor(buffer_size=config.sensor_buffer_size)
    asyncio.run(main_loop(config, sensor, debug))
        

if __name__ == "__main__":
    generate_synthetic()
    # generate_arduino()