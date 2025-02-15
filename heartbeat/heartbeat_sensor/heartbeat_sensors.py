from abc import ABC, abstractmethod
import serial
import time
import math
import random
from typing import List
import numpy as np


class HeartbeatSensor(ABC):
    """Abstract base class for heartbeat sensors."""

    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.current_bpm = 0.0
        self.signal_buffer: List[float] = []
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
            time.sleep(1.0 / sensor.sampling_rate)
    except KeyboardInterrupt:
        print("\nStopping sensor reading")
