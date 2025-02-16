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
        ax.set_ylim(600, 1200)
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
