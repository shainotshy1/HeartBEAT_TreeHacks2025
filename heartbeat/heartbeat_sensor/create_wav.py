import numpy as np
import wave
import struct
import time
import serial
from scipy.io.wavfile import write
from heartbeat_sensors import ArduinoHeartbeatSensor, SimulatedHeartbeatSensor

class HeartbeatToWAV:
    def __init__(self, sensor, duration=5, filename="heartbeat.wav", sampling_rate=1000):
        self.sensor = sensor
        self.duration = duration
        self.filename = filename
        self.sampling_rate = sampling_rate  # Audio sampling rate (1000 Hz for better quality)
        self.signal_buffer = []

    def record_signal(self):
        """Records heartbeat signal for a given duration."""
        num_samples = self.duration * self.sampling_rate
        print(f"Recording {self.duration} seconds of heartbeat data...")

        for _ in range(num_samples):
            signal = self.sensor.read_signal()
            self.signal_buffer.append(signal)
            time.sleep(1 / self.sampling_rate)  # Wait based on the sample rate

        print("Recording complete.")

    def normalize_signal(self):
        """Normalizes the heartbeat signal to fit 16-bit PCM range."""
        signal_array = np.array(self.signal_buffer)
        
        # Normalize to fit in 16-bit PCM range (-32768 to 32767)
        signal_array -= np.mean(signal_array)  # Center around zero
        signal_array /= np.max(np.abs(signal_array))  # Scale to [-1, 1]
        signal_array *= 32767  # Scale to 16-bit range
        signal_array = signal_array.astype(np.int16)

        return signal_array

    def save_to_wav(self):
        """Saves the normalized heartbeat signal as a WAV file."""
        audio_signal = self.normalize_signal()
        write(self.filename, self.sampling_rate, audio_signal)
        print(f"Heartbeat signal saved as {self.filename}")

# -------- Example Usage --------

# Use Arduino sensor if connected, else fallback to simulation
try:
    sensor = ArduinoHeartbeatSensor(serial_port="COM3")  # Adjust for your Arduino port
    print("Using Arduino sensor...")
except:
    sensor = SimulatedHeartbeatSensor()
    print("Using simulated sensor...")

# Create WAV file from heartbeat data
heartbeat_recorder = HeartbeatToWAV(sensor, duration=5, filename="heartbeat.wav", sampling_rate=1000)
heartbeat_recorder.record_signal()
heartbeat_recorder.save_to_wav()
