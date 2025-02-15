from abc import ABC, abstractmethod

import serial
import time
import math
import random
from typing import List

import numpy as np
import heartpy as hp  # Import heartpy

class HeartbeatSensor(ABC):
    """Abstract base class for heartbeat sensors."""

    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.current_bpm = 0.0
        self.signal_buffer: List[float] = []
        self.ibi_buffer: List[float] = []  # Inter-beat interval buffer
        self.last_peak_time: float = None  # Keep track of the last peak time
        self.buffer_size = sampling_rate * 5  # Store 5 seconds of data

    @abstractmethod
    def read_signal(self) -> float:
        """Read and return the latest heartbeat signal."""
        pass

    def calculate_bpm(self) -> float:
        """Calculate BPM from the signal buffer using peak detection."""
        if len(self.signal_buffer) < self.sampling_rate:
            return self.current_bpm

        # Simple peak detection
        peaks = []
        for i in range(1, len(self.signal_buffer) - 1):
            if (
                self.signal_buffer[i] > self.signal_buffer[i - 1]
                and self.signal_buffer[i] > self.signal_buffer[i + 1]
            ):
                peaks.append(i)

        if len(peaks) < 2:
            return self.current_bpm

        # Calculate average time between peaks
        peak_intervals = np.diff(peaks)
        avg_interval = np.mean(peak_intervals)

        # Convert to BPM
        if avg_interval > 0:
            self.current_bpm = 60.0 * self.sampling_rate / avg_interval
            return self.current_bpm

        # Update IBI Buffer
        current_time = time.time()
        if self.last_peak_time is not None:
            ibi = current_time - self.last_peak_time
            self.ibi_buffer.append(ibi)
            if len(self.ibi_buffer) > self.sampling_rate * 5:
                self.ibi_buffer.pop(0)  # Keep buffer size consistent
        self.last_peak_time = current_time  # Update last peak time

        return self.current_bpm

    def determine_emotion(self, hrv_features):
        """
        Determine emotion based on HRV features using simple heuristic rules.
        Higher HRV generally indicates positive emotional states, while lower HRV often indicates stress.
        """
        rmssd = hrv_features.get('rmssd', 0)  # Root mean square of successive RR interval differences
        sdnn = hrv_features.get('sdnn', 0)    # Standard deviation of NN intervals
        
        # Simple heuristic rules
        if rmssd > 50 and sdnn > 100:
            return "Calm/Relaxed"
        elif rmssd > 40 and sdnn > 80:
            return "Happy"
        elif rmssd < 20 and sdnn < 50:
            return "Stressed"
        elif rmssd < 30 and sdnn < 70:
            return "Anxious"
        elif 30 <= rmssd <= 40 and 70 <= sdnn <= 80:
            return "Neutral"
        else:
            return "Mixed"

    def analyze_hrv(self, rr_intervals):
        try:
            # Calculate basic HRV features
            if len(rr_intervals) < 2:
                return "Insufficient data"
                
            # Calculate RMSSD
            rr_diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(np.square(rr_diffs)))
            
            # Calculate SDNN
            sdnn = np.std(rr_intervals)
            
            hrv_features = {
                'rmssd': rmssd,
                'sdnn': sdnn
            }
            
            emotion = self.determine_emotion(hrv_features)
            return emotion
        except Exception as e:
            print(f"HRV analysis error: {e}")
            return "Error"


class ArduinoHeartbeatSensor(HeartbeatSensor):
    """Reads real heartbeat data from an Arduino device."""

    def __init__(
        self, serial_port: str, baud_rate: int = 115200, sampling_rate: int = 100
    ):
        super().__init__(sampling_rate)
        self.serial_port = serial_port
        self.serial = serial.Serial(serial_port, baud_rate)
        time.sleep(2)  # Wait for Arduino to reset

    def read_signal(self) -> float:
        """Read and parse serial data from Arduino."""
        try:
            if self.serial.in_waiting:
                line = self.serial.readline().decode("utf-8").strip()
                if line.startswith("Signal:"):
                    signal = float(line.split(":")[1])
                    self.signal_buffer.append(signal)
                    if len(self.signal_buffer) > self.buffer_size:
                        self.signal_buffer.pop(0)
                    return signal
                elif line.startswith("BPM:"):
                    self.current_bpm = float(line.split(":")[1])
                    return 0.0
        except Exception as e:
            print(f"Error reading from Arduino: {e}")
            return 0.0

    def __del__(self):
        """Clean up serial connection."""
        if hasattr(self, "serial"):
            self.serial.close()


class SimulatedHeartbeatSensor(HeartbeatSensor):
    """Generates synthetic heartbeat data for testing."""

    def __init__(
        self, base_bpm: float = 70.0, variability: float = 0.1, sampling_rate: int = 100
    ):
        super().__init__(sampling_rate)
        self.base_bpm = base_bpm
        self.variability = variability
        self.phase = 0.0

    def read_signal(self) -> float:
        """Generate synthetic heartbeat waveform."""
        # Calculate frequency from BPM
        frequency = self.base_bpm / 60.0

        # Generate basic sine wave
        t = time.time()
        self.phase += 2 * math.pi * frequency / self.sampling_rate

        # Add some randomness to simulate natural heart rate variability
        noise = random.gauss(0, self.variability)

        signal = math.sin(self.phase) + 0.3 * math.sin(2 * self.phase)  # Add harmonic
        signal = signal + noise

        self.signal_buffer.append(signal)

        # Simulate peak detection to get inter-beat intervals
        if signal > 0.9 and (len(self.signal_buffer) > 1 and self.signal_buffer[-2] < signal):
            current_time = time.time()
            if self.last_peak_time is not None:
                ibi = current_time - self.last_peak_time
                self.ibi_buffer.append(ibi)
                if len(self.ibi_buffer) > self.sampling_rate * 5:
                    self.ibi_buffer.pop(0)
            self.last_peak_time = current_time

        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        return signal

    def calculate_bpm(self) -> float:
        """Simulate BPM fluctuations."""
        # Add random walk to base_bpm
        self.base_bpm += random.gauss(0, self.variability)

        # Keep BPM in reasonable range
        self.base_bpm = max(40, min(200, self.base_bpm))

        self.current_bpm = self.base_bpm
        return self.current_bpm


# Example usage
if __name__ == "__main__":
    # For Arduino sensor
    try:
        sensor = ArduinoHeartbeatSensor("/dev/ttyUSB0")  # Adjust port as needed
        print("Using Arduino sensor")
    except:
        # Fallback to simulation if Arduino not available
        sensor = SimulatedHeartbeatSensor()
        print("Using simulated sensor")

    try:
        while True:
            signal = sensor.read_signal()
            bpm = sensor.calculate_bpm()

            if bpm > 0:
                print(f"Signal: {signal:.2f}, BPM: {bpm:.1f}")

            # Analyze HRV using HeartPy if enough IBI data is available
            if len(sensor.ibi_buffer) > 60:  # Requires enough data points
                try:
                    wd, m = hp.process_segmentwise(np.array(sensor.ibi_buffer), sample_rate=sensor.sampling_rate, segment_width=60, segment_overlap=0.5)  # Adjust segment_width and overlap as needed
                    print("HeartPy analysis results:", m)

                    # Here you would use the m dictionary to classify valence/arousal
                    # This part requires a trained model or heuristic rules based on HRV features.

                    emotion = sensor.analyze_hrv(np.array(sensor.ibi_buffer))
                    print(f"Emotion: {emotion}")
                except Exception as e:
                    print(f"HeartPy analysis error: {e}")

            time.sleep(1.0 / sensor.sampling_rate)

    except KeyboardInterrupt:
        print("\nStopping sensor reading")
