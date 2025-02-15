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
        Determine emotion based on HRV features using research-based thresholds.
        References:
        - Task Force of The European Society of Cardiology, 1996
        - Kim et al., 2018 (Emotion Classification Using HRV Parameters)
        - Shaffer & Ginsberg, 2017 (HRV Metrics and Norms)
        """
        rmssd = hrv_features.get('rmssd', 0)  # ms
        sdnn = hrv_features.get('sdnn', 0)    # ms
        
        # Get additional HRV features if available
        hr = hrv_features.get('bpm', 70)      # beats per minute
        
        # Normalize RMSSD and SDNN based on individual baseline
        # (typically collected during neutral state)
        baseline_rmssd = 27.0  # Typical baseline for healthy adults
        baseline_sdnn = 50.0   # Typical baseline for healthy adults
        
        rel_rmssd = rmssd / baseline_rmssd
        rel_sdnn = sdnn / baseline_sdnn
        
        # Define emotional state thresholds
        if hr > 100 and rmssd < 20 and sdnn < 30:
            return "High Stress/Fear"
        elif hr > 90 and rmssd < 25 and sdnn < 40:
            return "Anxious"
        elif (rel_rmssd > 1.2 and rel_sdnn > 1.2) and (hr < 75):
            if rel_rmssd > 1.5 and rel_sdnn > 1.5:
                return "Deep Relaxation"
            return "Calm"
        elif (rel_rmssd > 1.1 and rel_sdnn > 1.1) and (75 <= hr <= 85):
            return "Happy/Excited"
        elif (0.8 <= rel_rmssd <= 1.2) and (0.8 <= rel_sdnn <= 1.2):
            if 65 <= hr <= 75:
                return "Neutral"
            elif hr > 75:
                return "Mild Arousal"
            else:
                return "Mild Relaxation"
        elif rmssd < (0.8 * baseline_rmssd) and sdnn < (0.8 * baseline_sdnn):
            if hr > 80:
                return "Mild Stress"
            else:
                return "Focus/Concentration"
        
        # Calculate complexity score for mixed emotional states
        complexity = abs((rel_rmssd - 1.0) * (rel_sdnn - 1.0))
        if complexity > 0.3:
            return "Mixed Emotional State"
            
        return "Neutral"


    def analyze_hrv(self, rr_intervals):
        try:
            # Add debug print
            print(f"RR intervals received: {rr_intervals}")
            
            # Check for valid data - fixed condition
            if rr_intervals is None or len(rr_intervals) < 2:
                print("Insufficient RR interval data")
                return "Stressed"  # Default to stressed when data is insufficient
                
            # Calculate HRV features
            rr_diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(np.square(rr_diffs)))
            sdnn = np.std(rr_intervals)
            
            print(f"Calculated RMSSD: {rmssd}, SDNN: {sdnn}")  # Debug print
            
            hrv_features = {
                'rmssd': rmssd,
                'sdnn': sdnn
            }
            
            emotion = self.determine_emotion(hrv_features)
            return emotion
        except Exception as e:
            print(f"Error in analyze_hrv: {str(e)}")
            return "Stressed"  # Default to stressed when there's an error

    def process_data(self, data):
        try:
            # Basic data validation
            print(f"Data length: {len(data)}")
            print(f"Data range: min={min(data)}, max={max(data)}")
            print(f"Sample rate being used: {self.sampling_rate} Hz")
            
            # Check if we have enough data
            min_required_duration = 10  # seconds
            min_required_samples = min_required_duration * self.sampling_rate
            if len(data) < min_required_samples:
                print(f"Warning: Data length ({len(data)}) is less than minimum required ({min_required_samples})")
            
            # Check for signal quality
            signal_range = max(data) - min(data)
            if signal_range < 100:  # Arbitrary threshold, adjust based on your sensor
                print(f"Warning: Low signal variation ({signal_range})")
            
            # Process with more detailed error handling
            data = np.array(data, dtype='float32')
            filtered = hp.filter_signal(data, 
                                      cutoff=[0.75, 3.5],
                                      sample_rate=self.sampling_rate, 
                                      order=2,
                                      filtertype='bandpass')
            
            try:
                wd, m = hp.process(filtered, 
                                 sample_rate=self.sampling_rate,
                                 high_precision=True,
                                 clean_rr=True)
                print("HeartPy processing completed successfully")
            except Exception as hp_error:
                print(f"HeartPy processing error: {str(hp_error)}")
                m = {}
            
            if not m:
                print("HeartPy analysis produced no measures")
                return {
                    "bpm": np.mean(60 / np.diff(self.ibi_buffer)) if hasattr(self, 'ibi_buffer') else 0,
                    "ibi_buffer": self.ibi_buffer if hasattr(self, 'ibi_buffer') else [],
                    "emotion": self.analyze_hrv(self.ibi_buffer) if hasattr(self, 'ibi_buffer') else "Unknown",
                    "signal_quality_warning": "Low signal quality or insufficient data"
                }
            
            print(f"HeartPy measures: {m}")
            return m
        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            return {}


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

    def __init__(self):
        self.SAMPLE_RATE = 100  # Hz
        self.duration = 15  # 15 seconds
        self.phase = 0  # Initialize phase
        self.sampling_rate = self.SAMPLE_RATE  # Add sampling_rate attribute
        super().__init__(self.SAMPLE_RATE)
        self.current_bpm = 70.0  # Assuming a default base_bpm
        self.signal_buffer = []
        self.ibi_buffer = []
        self.last_peak_time = None
        self.buffer_size = self.SAMPLE_RATE * self.duration

    def read_data(self):
        # Generate time points
        t = np.linspace(0, self.duration, int(self.duration * self.SAMPLE_RATE))
        
        # Base heart rate (beats per second)
        heart_rate = 1.2  # ~72 BPM
        
        # Generate simulated heartbeat signal
        signal = np.zeros_like(t)
        for i in range(int(self.duration * heart_rate)):
            beat_time = i / heart_rate
            # Add a heartbeat pulse at each beat time
            pulse = np.exp(-(t - beat_time)**2 * 30) * np.sin(2 * np.pi * 7 * (t - beat_time))
            signal += pulse
            
        # Add some noise
        noise = np.random.normal(0, 0.1, len(t))
        signal += noise
        
        return signal.tolist()

    def read_signal(self) -> float:
        """Generate synthetic heartbeat waveform."""
        return self.read_data()  # Use our existing read_data method

    def calculate_bpm(self) -> float:
        """Simulate BPM fluctuations."""
        # Add random walk to base_bpm
        self.current_bpm += random.gauss(0, 0.1)

        # Keep BPM in reasonable range
        self.current_bpm = max(40, min(200, self.current_bpm))

        return self.current_bpm


# Example usage
if __name__ == "__main__":
    try:
        sensor = SimulatedHeartbeatSensor()
        print("Using simulated sensor")
        
        while True:
            signal = sensor.read_data()
            signal_array = np.array(signal)
            
            try:
                # First filter the signal
                filtered = hp.filter_signal(signal_array, 
                                         cutoff=[0.75, 3.5],
                                         sample_rate=sensor.SAMPLE_RATE, 
                                         order=2,
                                         filtertype='bandpass')
                
                # Process with correct parameters
                working_data, measures = hp.process(filtered, 
                                                  sample_rate=sensor.SAMPLE_RATE,
                                                  high_precision=True)
                
                # Convert peaklist to numpy array and check length
                peaks = np.array(working_data['peaklist'])
                
                if len(peaks) >= 2:  # Need at least 2 peaks
                    # Calculate RR intervals from peaks
                    rr_intervals = np.diff(peaks) / sensor.SAMPLE_RATE * 1000
                    
                    # Remove negative and zero values
                    rr_intervals = rr_intervals[rr_intervals > 0]
                    
                    # Remove outliers (RR intervals should typically be between 300-1200ms)
                    rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 1200)]
                    
                    if len(rr_intervals) >= 2:
                        # Calculate BPM
                        bpm = 60000 / np.mean(rr_intervals)
                        bpm = np.clip(bpm, 60, 100)
                        
                        # Get emotion
                        emotion = sensor.analyze_hrv(rr_intervals)
                        
                        print(f"\rBPM: {bpm:.1f} | Emotion: {emotion} | RR Intervals: {len(rr_intervals)}", end='', flush=True)
                    else:
                        print("\rInsufficient valid RR intervals...", end='', flush=True)
                else:
                    print("\rCollecting data...", end='', flush=True)
                
            except Exception as e:
                print(f"\rProcessing error: {str(e)}", end='', flush=True)
                continue
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping sensor reading...")
    except Exception as e:
        print(f"\nError: {str(e)}")
