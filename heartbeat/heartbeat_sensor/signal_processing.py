from typing import List
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import variation
from scipy.ndimage import gaussian_filter1d

class SignalProcessor:
    def __init__(self, window_size: int = 500, filter_update_interval: int = 50):
        """
        Initialize the signal processor.
        
        Args:
            window_size: Number of samples to store in history
            filter_update_interval: Number of new samples before recomputing filter
        """
        self.window_size = window_size
        self.filter_update_interval = filter_update_interval
        self.raw_signal: List[float] = []
        self.filtered_signal: List[float] = []
        self.sampling_rate = 100  # Hz, assumed from sensor
        self.samples_since_filter = 0
        
    def filter_noise_butterworth(self, signal: List[float] = None) -> List[float]:
        """
        Applies a Butterworth low-pass filter to remove noise from the signal.
        If no signal is provided, uses the internal raw_signal buffer.
        
        Args:
            signal: Optional raw heartbeat signal data
            
        Returns:
            Filtered signal data
        """
        if signal is None:
            signal = self.raw_signal
            
        if len(signal) < 20:  # Not enough samples for proper filtering
            return signal.copy()
            
        # Convert to numpy array for processing
        signal_array = np.array(signal)
        
        # Design Butterworth filter
        nyquist = self.sampling_rate * 0.5
        cutoff = 8.0  # Hz, typical for heartbeat signals
        order = 2  # Reduced order for better stability with short signals
        b, a = butter(order, cutoff / nyquist, btype='low')
        
        # Apply filter with minimal padding
        filtered_signal = filtfilt(b, a, signal_array, padlen=min(3*order, len(signal)-1))
        
        return filtered_signal.tolist()

    def get_filtered_value_butterworth(self) -> float:
        """
        Get the most recent filtered value. Updates filter only when needed.
        
        Returns:
            Most recent filtered value
        """
        if not self.raw_signal:
            return 0.0
            
        # Update filter if enough new samples have accumulated
        if self.samples_since_filter >= self.filter_update_interval:
            self.filtered_signal = self.filter_noise_butterworth()
            self.samples_since_filter = 0
            
        # Return the most recent filtered value
        if self.filtered_signal:
            return self.filtered_signal[-1]
        else:
            # If we haven't computed any filtered values yet, filter the whole signal
            self.filtered_signal = self.filter_noise_butterworth()
            return self.filtered_signal[-1]

    def filter_noise_ema(self, signal: List[float] = None, alpha: float = 0.1) -> List[float]:
        """
        Applies Exponential Moving Average filter to the signal.
        
        Args:
            signal: Optional raw heartbeat signal data. If None, uses internal buffer
            alpha: Smoothing factor (0-1). Lower values = more smoothing
            
        Returns:
            Filtered signal data of same length as input
        """
        if signal is None:
            signal = self.raw_signal
            
        if not signal:
            return []
            
        # Convert to numpy array for processing
        signal_array = np.array(signal)
        filtered = np.zeros_like(signal_array)
        
        # Initialize with first value
        filtered[0] = signal_array[0]
        
        # Apply EMA filter
        for i in range(1, len(signal_array)):
            filtered[i] = alpha * signal_array[i] + (1 - alpha) * filtered[i-1]
            
        return filtered.tolist()

    def get_filtered_value_ema(self, alpha: float = 0.1) -> float:
        """
        Get the most recent EMA filtered value.
        
        Args:
            alpha: Smoothing factor for EMA filter
            
        Returns:
            Most recent filtered value
        """
        if not self.raw_signal:
            return 0.0
            
        # Only need to update the last value using EMA
        if not self.filtered_signal:
            self.filtered_signal = self.filter_noise_ema(alpha=alpha)
        else:
            # Update just the last value using EMA formula
            new_value = alpha * self.raw_signal[-1] + (1 - alpha) * self.filtered_signal[-1]
            self.filtered_signal.append(new_value)
            if len(self.filtered_signal) > self.window_size:
                self.filtered_signal.pop(0)
                
        return self.filtered_signal[-1]
    
    def detect_peaks(self, signal: List[float]) -> List[int]:
        """
        Identifies peaks in the heartbeat signal that correspond to heartbeats.
        
        Args:
            signal: Filtered heartbeat signal data
        """
        peaks, _ = find_peaks(gaussian_filter1d(signal, sigma=5), height=0, distance=30)
        return peaks.tolist()

    def detect_peaks_old(self, signal: List[float]) -> List[int]:
        """
        Identifies peaks in the heartbeat signal that correspond to heartbeats.
        
        Args:
            signal: Filtered heartbeat signal data
            
        Returns:
            List of indices where peaks were detected
        """
        # Convert to numpy array
        signal_array = np.array(signal)
        
        # Find peaks using local maxima
        peaks = []
        for i in range(1, len(signal_array) - 1):
            if (signal_array[i] > signal_array[i-20] and 
                signal_array[i] > signal_array[i+20] and
                signal_array[i] > np.mean(signal_array[max(0,i-30):min(len(signal_array),i+30)])):
                peaks.append(i)
        
        return peaks
    
    def update_signal(self, new_data: float):
        """
        Updates the signal buffer with new data.
        
        Args:
            new_data: New signal value to add
        """
        self.raw_signal.append(new_data)
        if len(self.raw_signal) > self.window_size:
            self.raw_signal.pop(0)
            
        self.samples_since_filter += 1
