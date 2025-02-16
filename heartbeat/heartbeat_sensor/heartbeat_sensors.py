from abc import ABC, abstractmethod

import serial
import time
import math
import random
from typing import List, Tuple, Dict
import numpy as np
import heartpy as hp
from heartbeat.heartbeat_sensor.emotion import Emotion

class HeartbeatSensor(ABC):
    """Abstract base class for heartbeat sensors."""

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size   # base agg stats off the last n data points
        self.signal_buffer = []
        self.working_data = None
        self.measures = None

    @abstractmethod
    def read_signal(self) -> Tuple[np.str_, float] | None:
        """Read and return the latest heartbeat signal and timestamp."""
        pass
    
    def process(self, signal=None, timestamps=None, timing=False):
        if signal is None:
            signal = self.signal_values
            timestamps = self.timestamps
        signal = np.array(signal)
        timestamps = np.array(timestamps)
        sample_rate = hp.get_samplerate_datetime(timestamps, timeformat='%Y-%m-%d %H:%M:%S.%f')
        t0 = time.time()
        self.working_data, self.measures = hp.process(signal, sample_rate, report_time = True)
        t1 = time.time()
        if timing:
            print(f"Processing took {t1-t0} seconds")
        
    def add_to_buffer(self, signal, time):
        if len(self.signal_buffer) >= self.buffer_size:
            self.signal_buffer.pop(0)
        self.signal_buffer.append((signal, time))

    def determine_emotion(self):
        """
        Determine emotion based on HRV features using research-based thresholds.
        """
        if self.measures is None:
            return None
        bpm = self.measures['bpm']
        rmssd = self.measures['rmssd']
        sdnn = self.measures['sdnn']
        pnn50 = self.measures['pnn50']
        sd1 = self.measures['sd1']
        sd2 = self.measures['sd2']
        breathing_rate = self.measures['breathingrate']
        
        # Normalize parameters based on individual baseline
        # (These should be collected during a neutral state)
        baseline_bpm = 70  # Example baseline, should be personalized
        baseline_rmssd = 27.0
        baseline_sdnn = 50.0
        baseline_pnn50 = 10.0
        baseline_sd1 = 30.0
        baseline_sd2 = 70.0
        baseline_breathing_rate = 15.0
        
        rel_bpm = bpm / baseline_bpm
        rel_rmssd = rmssd / baseline_rmssd
        rel_sdnn = sdnn / baseline_sdnn
        rel_pnn50 = pnn50 / baseline_pnn50
        rel_sd1 = sd1 / baseline_sd1
        rel_sd2 = sd2 / baseline_sd2
        rel_breathing_rate = breathing_rate / baseline_breathing_rate
        
        # Define emotional state thresholds
        if rel_bpm > 1.3 and rel_rmssd < 0.7 and rel_sdnn < 0.7 and rel_pnn50 < 0.7 and rel_breathing_rate > 1.2:
            return Emotion.HIGH_STRESS_FEAR
        elif rel_bpm > 1.2 and rel_rmssd < 0.8 and rel_sdnn < 0.8 and rel_pnn50 < 0.8 and rel_breathing_rate > 1.1:
            return Emotion.ANXIOUS
        elif (rel_rmssd > 1.2 and rel_sdnn > 1.2 and rel_pnn50 > 1.2) and (rel_bpm < 0.9) and (rel_breathing_rate < 0.9):
            if rel_rmssd > 1.5 and rel_sdnn > 1.5 and rel_pnn50 > 1.5:
                return Emotion.DEEP_RELAXATION
            return Emotion.CALM
        elif (rel_rmssd > 1.1 and rel_sdnn > 1.1 and rel_pnn50 > 1.1) and (1.0 <= rel_bpm <= 1.2):
            return Emotion.HAPPY_EXCITED
        elif (0.9 <= rel_rmssd <= 1.1) and (0.9 <= rel_sdnn <= 1.1) and (0.9 <= rel_pnn50 <= 1.1):
            if 0.95 <= rel_bpm <= 1.05:
                return Emotion.NEUTRAL
            elif rel_bpm > 1.05:
                return Emotion.MILD_AROUSAL
            else:
                return Emotion.MILD_RELAXATION
        elif rel_rmssd < 0.9 and rel_sdnn < 0.9 and rel_pnn50 < 0.9:
            if rel_bpm > 1.1:
                return Emotion.MILD_STRESS
            else:
                return Emotion.FOCUS_CONCENTRATION
        
        # Use SD1/SD2 ratio for additional insight
        sd1_sd2_ratio = sd1 / sd2
        baseline_sd1_sd2_ratio = baseline_sd1 / baseline_sd2
        rel_sd1_sd2_ratio = sd1_sd2_ratio / baseline_sd1_sd2_ratio
        
        if rel_sd1_sd2_ratio > 1.2:
            return Emotion.INCREASED_PARASYMPATHETIC_ACTIVITY
        elif rel_sd1_sd2_ratio < 0.8:
            return Emotion.INCREASED_SYMPATHETIC_ACTIVITY
        
        return Emotion.MIXED_EMOTIONAL_STATE

    @property
    def signal_values(self):
        return [signal for signal, _ in self.signal_buffer]
    
    @property
    def timestamps(self):
        return [timestamp for _, timestamp in self.signal_buffer]


class ArduinoHeartbeatSensor(HeartbeatSensor):
    """Reads real heartbeat data from an Arduino device."""

    def __init__(
        self, serial_port: str, baud_rate: int = 115200, buffer_size: int = 1000, num_steps: int = 1000, warmup_steps: int = 1000
    ):
        super().__init__(buffer_size)
        self.num_steps = num_steps
        self.serial_port = serial_port
        self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset

    def read_signal(self) -> Tuple[np.str_, float] | None:
        """
        Read and parse serial data from Arduino.
        """
        # data_points = []
        import tqdm
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x = []
        y = []
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1200)
        line, = ax.plot([], [], lw=2)

        # for i in range(self.num_steps):
        for i in tqdm.tqdm(range(self.num_steps)):
            time.sleep(0.01) 
            data = self.arduino.readline().decode('utf-8').strip()
            if not data:
                continue
            data = float(data)
            self.add_to_buffer(data, time.time())

            x.append(i)
            y.append(data)
            # line.set_data(x, y)
            line.set_xdata(x)
            line.set_ydata(y)
            plt.draw()
            plt.pause(1e-17)
            # data_points.append(data)
        # try:
        #     if self.arduino.in_waiting:
        #         line = self.arduino.readline().decode("utf-8").strip()
        #         signal = float(line)
        #         timestamp = time.time()
        #         self.add_to_buffer(signal, timestamp)
        #         return signal, timestamp
        #     return None
        # except Exception as e:
        #     print(f"Error reading from Arduino: {e}")
        #     return None

    def __del__(self):
        """Clean up serial connection."""
        if hasattr(self, "serial"):
            self.arduino.close()

            
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
