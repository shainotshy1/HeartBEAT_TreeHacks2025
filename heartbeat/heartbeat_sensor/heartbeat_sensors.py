from abc import ABC, abstractmethod
import serial
import time
import math
import random
from typing import List, Tuple, Dict
import numpy as np
import heartpy as hp


class HeartbeatSensor(ABC):
    """Abstract base class for heartbeat sensors."""

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size   # base agg stats off the last n data points
        self.signal_buffer = []
        self.working_data = None
        self.measures = None

    @abstractmethod
    def read_signal(self) -> float:
        """Read and return the latest heartbeat signal."""
        pass
    
    def process(self, timing=False):
        data, timer = zip(*self.signal_buffer)
        data = np.array(data)
        timer = np.array(timer)
        sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')
        t0 = time.time()
        self.working_data, self.measures = hp.process(data, sample_rate, report_time = True)
        t1 = time.time()
        if timing:
            print(f"Processing took {t1-t0} seconds")
        
    def add_to_buffer(self, signal, time):
        if len(self.signal_buffer) >= self.buffer_size:
            self.signal_buffer.pop(0)
        self.signal_buffer.append((signal, time))
    
    def get_bpm(self) -> float | None:
        if self.measures is None:
            return None
        return self.measures['bpm']
    
    def get_hrv(self) -> float | None:
        if self.measures is None:
            return None
        return self.measures['rmssd']


class ArduinoHeartbeatSensor(HeartbeatSensor):
    """Reads real heartbeat data from an Arduino device."""

    def __init__(
        self, serial_port: str, baud_rate: int = 115200, buffer_size: int = 1000
    ):
        super().__init__(buffer_size)
        self.serial_port = serial_port
        self.serial = serial.Serial(serial_port, baud_rate)
        time.sleep(2)  # Wait for Arduino to reset

    def read_signal(self) -> Tuple[np.str_, float] | None:
        """
        Read and parse serial data from Arduino.
        """
        try:
            if self.serial.in_waiting:
                line = self.serial.readline().decode("utf-8").strip()
                signal = float(line)
                timestamp = time.time()
                self.add_to_buffer(signal, timestamp)
                return signal, timestamp
            return None
        except Exception as e:
            print(f"Error reading from Arduino: {e}")
            return None

    def __del__(self):
        """Clean up serial connection."""
        if hasattr(self, "serial"):
            self.serial.close()

            
class SimulatedHeartbeatSensor(HeartbeatSensor):   
    def __init__(self, buffer_size: int = 1000):
        super().__init__(buffer_size)
        self.data, self.times = hp.load_exampledata(2)
        self.length = len(self.data)
        self.index = 0
    
    def read_signal(self) -> Tuple[np.str_, float] | None:
        if self.index >= self.length:
            return None
        signal = self.data[self.index]
        timestamp = self.times[self.index]
        self.index += 1
        self.add_to_buffer(signal, timestamp)
        return signal, timestamp


# Example usage
# if __name__ == "__main__":
#     # For Arduino sensor
#     try:
#         sensor = ArduinoHeartbeatSensor("/dev/ttyUSB0")  # Adjust port as needed
#         print("Using Arduino sensor")
#     except:
#         # Fallback to simulation if Arduino not available
#         sensor = SimulatedHeartbeatSensor()
#         print("Using simulated sensor")

#     try:
#         while True:
#             signal = sensor.read_signal()
#             bpm = sensor.calculate_bpm()
#             if bpm > 0:
#                 print(f"Signal: {signal:.2f}, BPM: {bpm:.1f}")
#             time.sleep(1.0 / sensor.sampling_rate)
#     except KeyboardInterrupt:
#         print("\nStopping sensor reading")
