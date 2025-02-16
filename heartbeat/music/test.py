import pygame
from pygame.locals import *
import math
import numpy
import numpy as np

size = (1366, 720)

bits = 16
# The number of channels specified here is NOT the channels talked about here http://www.pygame.org/docs/ref/mixer.html#pygame.mixer.get_num_channels

pygame.mixer.pre_init(44100, -bits, 2)
pygame.init()
_display_surf = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)

# Frequency (in Hz) and duration (in seconds) for the series of notes
# notes = [
#     (440, 0.5),  # A4
#     (550, 0.5),  # C#5
#     (660, 0.5),  # E5
#     (880, 0.5),  # A5
#     (660, 0.5),  # E5
#     (550, 0.5),  # C#5
#     (440, 0.5),  # A4
# ]

notes = []

from heartbeat.heartbeat_sensor.heartbeat_sensors import ArduinoHeartbeatSensor
hs = ArduinoHeartbeatSensor(serial_port='/dev/cu.usbmodem1201', baud_rate=9600)

import time
time.sleep(1) 
# hs.read_signal()

import pickle
raw_signals = pickle.load(open('tests/raw_signals.pkl', 'rb'))

signals = [s for s, _ in raw_signals]
t = [t for _, t in raw_signals]

from heartbeat.heartbeat_sensor.signal_processing import SignalProcessor
np.random.seed(42)
processor = SignalProcessor(filter_update_interval=50)
filtered_signals = processor.filter_noise_ema(signals)

# def fm_synthesis(frequency, duration, mod_index=2, mod_freq=3):
#     sample_rate = 44100
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

#     # Carrier and modulator
#     carrier = np.sin(2 * np.pi * frequency * t)
#     modulator = np.sin(2 * np.pi * mod_freq * t)
    
#     # FM synthesis formula
#     sound = np.sin(2 * np.pi * frequency * t + mod_index * modulator)

#     # Convert sound to 16-bit audio
#     sound = np.int16(sound * 32767)  # scale to 16-bit PCM range
#     sound = pygame.sndarray.make_sound(sound)
#     return sound

def fm_synthesis(frequency, duration, mod_index=2, mod_freq=3):
    frequency_right = frequency
    frequency_left = frequency
    sample_rate = 44100  # 44.1 kHz sample rate
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time array
    
    # Carrier and modulator for the left channel
    carrier_left = np.sin(2 * np.pi * frequency_left * t)
    modulator_left = np.sin(2 * np.pi * mod_freq * t)
    sound_left = np.sin(2 * np.pi * frequency_left * t + mod_index * modulator_left)
    
    # Carrier and modulator for the right channel
    carrier_right = np.sin(2 * np.pi * frequency_right * t)
    modulator_right = np.sin(2 * np.pi * mod_freq * t)
    sound_right = np.sin(2 * np.pi * frequency_right * t + mod_index * modulator_right)
    
    # Combine both channels into a stereo signal
    stereo_sound = np.stack((sound_left, sound_right), axis=-1)
    
    # Normalize and convert to 16-bit PCM format
    stereo_sound = np.int16(stereo_sound * 32767)
    stereo_sound = pygame.sndarray.make_sound(stereo_sound)
    return stereo_sound

two_octave_c_major = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50]

# play the notes in the two octave c major scale
# for note in two_octave_c_major:
#     notes.append((note, 0.5))


# mean_signal = numpy.mean(hs.signal_buffer)
# median_signal = numpy.median(hs.signal_buffer)
# signal_buffer = [signal[0] for signal in hs.signal_buffer]
# range_signal = max(signal_buffer) - min(signal_buffer)

import tqdm
# for signal in hs.signal_buffer:
# for signal in tqdm.tqdm(hs.signal_buffer):
#     # frequency = 440 + 220 * (signal[0] - median_signal)
#     frequency = (signal[0] - min(signal_buffer)) / range_signal
#     idx = int(frequency * len(two_octave_c_major) - 1)
#     frequency = two_octave_c_major[idx]
#     notes.append((frequency, 0.5))
#     # notes.append((frequency, 0.5))
#     # notes.append((normalized, 0.5))
#     # notes.append((signal[0], 0.5))

range_signal = max(filtered_signals) - min(filtered_signals)
for signal in tqdm.tqdm(filtered_signals):
    frequency = (signal - min(filtered_signals)) / range_signal
    idx = int(frequency * len(two_octave_c_major) - 1)
    frequency = two_octave_c_major[idx]
    notes.append((frequency, 0.25))

sample_rate = 44100
max_sample = 2**(bits - 1) - 1

def generate_note(frequency, duration):
    """Generates a sound for a given frequency and duration."""
    n_samples = int(round(duration * sample_rate))
    buf = numpy.zeros((n_samples, 2), dtype=numpy.int16)

    for s in range(n_samples):
        t = float(s) / sample_rate
        buf[s][0] = int(round(max_sample * math.sin(2 * math.pi * frequency * t)))  # left channel
        buf[s][1] = int(round(max_sample * 0.5 * math.sin(2 * math.pi * frequency * t)))  # right channel

    sound = pygame.sndarray.make_sound(buf)
    return sound


# save the notes to a file
import pickle
with open('notes2.pkl', 'wb') as f:
    pickle.dump(notes, f)

# Loop through each note in the series and play it
# for frequency, duration in notes:
for frequency, duration in tqdm.tqdm(notes):
    # sound = generate_note(frequency, duration)
    sound = fm_synthesis(frequency, duration)
    sound.play()
    pygame.time.delay(int(duration * 1000))  # Delay for the duration of the note

# # Keep the window open until it is closed
# _running = True
# while _running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             _running = False
#             break

# pygame.quit()
