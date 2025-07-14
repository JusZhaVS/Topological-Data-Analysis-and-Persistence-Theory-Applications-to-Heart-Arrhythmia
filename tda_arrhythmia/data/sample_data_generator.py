#!/usr/bin/env python3
"""
Sample Data Generator for TDA Cardiac Analysis

This module generates synthetic ECG data for testing and demonstration purposes.
"""

import numpy as np
import pickle
import os
from typing import Tuple, List, Dict


class SyntheticECGGenerator:
    """Generate realistic synthetic ECG signals with various arrhythmias."""
    
    def __init__(self, fs: int = 250):
        """
        Initialize ECG generator.
        
        Parameters:
        -----------
        fs : int
            Sampling frequency in Hz
        """
        self.fs = fs
        
    def generate_normal_ecg(self, duration: float = 10.0, hr: float = 75.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate normal sinus rhythm ECG.
        
        Parameters:
        -----------
        duration : float
            Duration in seconds
        hr : float
            Heart rate in BPM
            
        Returns:
        --------
        t : np.ndarray
            Time vector
        signal : np.ndarray
            ECG signal
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        signal = np.zeros_like(t)
        
        # RR interval with physiological variability
        rr_base = 60.0 / hr
        beat_times = []
        current_time = 0
        
        while current_time < duration:
            # Heart rate variability (5% standard deviation)
            rr_interval = rr_base * np.random.normal(1.0, 0.05)
            current_time += rr_interval
            if current_time < duration:
                beat_times.append(current_time)
        
        # Generate QRS complexes and T waves
        for beat_time in beat_times:
            # P wave (atrial depolarization)
            p_wave = self._generate_p_wave(t, beat_time - 0.16)
            
            # QRS complex (ventricular depolarization)
            qrs_complex = self._generate_qrs_complex(t, beat_time)
            
            # T wave (ventricular repolarization)
            t_wave = self._generate_t_wave(t, beat_time + 0.32)
            
            signal += p_wave + qrs_complex + t_wave
        
        # Add realistic noise
        signal += self._add_physiological_noise(t, signal)
        
        return t, signal
    
    def generate_arrhythmic_ecg(self, arrhythmia_type: str, duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ECG with specific arrhythmia.
        
        Parameters:
        -----------
        arrhythmia_type : str
            Type of arrhythmia ('atrial_fibrillation', 'ventricular_tachycardia', 
            'premature_ventricular_contractions', 'supraventricular_tachycardia')
        duration : float
            Duration in seconds
            
        Returns:
        --------
        t : np.ndarray
            Time vector
        signal : np.ndarray
            ECG signal
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        signal = np.zeros_like(t)
        
        if arrhythmia_type == 'atrial_fibrillation':
            signal = self._generate_atrial_fibrillation(t)
        elif arrhythmia_type == 'ventricular_tachycardia':
            signal = self._generate_ventricular_tachycardia(t)
        elif arrhythmia_type == 'premature_ventricular_contractions':
            signal = self._generate_pvc_rhythm(t)
        elif arrhythmia_type == 'supraventricular_tachycardia':
            signal = self._generate_svt(t)
        else:
            raise ValueError(f"Unknown arrhythmia type: {arrhythmia_type}")
        
        # Add noise
        signal += self._add_physiological_noise(t, signal)
        
        return t, signal
    
    def _generate_p_wave(self, t: np.ndarray, center: float) -> np.ndarray:
        """Generate P wave."""
        if center < 0 or center > t[-1]:
            return np.zeros_like(t)
        
        duration = 0.08  # 80ms
        amplitude = 0.15
        
        wave = amplitude * np.exp(-0.5 * ((t - center) / (duration/4))**2)
        return wave
    
    def _generate_qrs_complex(self, t: np.ndarray, center: float, amplitude: float = 1.0, width_factor: float = 1.0) -> np.ndarray:
        """Generate QRS complex."""
        if center < 0 or center > t[-1]:
            return np.zeros_like(t)
        
        duration = 0.08 * width_factor  # Normal: 80ms
        
        # Q wave (small negative deflection)
        q_wave = -0.1 * amplitude * np.exp(-0.5 * ((t - center + 0.02) / 0.01)**2)
        
        # R wave (main positive deflection)
        r_wave = amplitude * np.exp(-0.5 * ((t - center) / (duration/6))**2)
        
        # S wave (negative deflection after R)
        s_wave = -0.3 * amplitude * np.exp(-0.5 * ((t - center - 0.02) / 0.015)**2)
        
        return q_wave + r_wave + s_wave
    
    def _generate_t_wave(self, t: np.ndarray, center: float) -> np.ndarray:
        """Generate T wave."""
        if center < 0 or center > t[-1]:
            return np.zeros_like(t)
        
        duration = 0.16  # 160ms
        amplitude = 0.3
        
        wave = amplitude * np.exp(-0.5 * ((t - center) / (duration/4))**2)
        return wave
    
    def _generate_atrial_fibrillation(self, t: np.ndarray) -> np.ndarray:
        """Generate atrial fibrillation pattern."""
        signal = np.zeros_like(t)
        
        # Irregular RR intervals (300-600ms)
        current_time = 0
        while current_time < t[-1]:
            rr_interval = np.random.uniform(0.3, 0.6)
            current_time += rr_interval
            
            if current_time < t[-1]:
                # Irregular QRS morphology
                amplitude = np.random.uniform(0.8, 1.2)
                width_factor = np.random.uniform(0.9, 1.1)
                
                qrs = self._generate_qrs_complex(t, current_time, amplitude, width_factor)
                signal += qrs
                
                # Sometimes add T wave
                if np.random.random() > 0.3:
                    t_wave = self._generate_t_wave(t, current_time + 0.25)
                    signal += t_wave
        
        # Add fibrillatory waves (irregular baseline)
        fibrillatory_waves = 0.05 * np.random.randn(len(t))
        # Apply high-frequency filtering to simulate fibrillatory activity
        from scipy import signal as sp_signal
        b, a = sp_signal.butter(4, [3, 15], btype='band', fs=self.fs)
        fibrillatory_waves = sp_signal.filtfilt(b, a, fibrillatory_waves)
        signal += fibrillatory_waves
        
        return signal
    
    def _generate_ventricular_tachycardia(self, t: np.ndarray) -> np.ndarray:
        """Generate ventricular tachycardia pattern."""
        signal = np.zeros_like(t)
        
        # Fast, regular rate (150-250 BPM)
        hr = np.random.uniform(150, 220)
        rr_interval = 60.0 / hr
        
        beat_times = np.arange(0, t[-1], rr_interval)
        
        for beat_time in beat_times:
            # Wide QRS complexes with abnormal morphology
            amplitude = np.random.uniform(1.5, 2.5)
            width_factor = np.random.uniform(1.5, 2.0)  # Wide QRS
            
            qrs = self._generate_qrs_complex(t, beat_time, amplitude, width_factor)
            signal += qrs
            
            # Abnormal T waves (sometimes inverted)
            t_amplitude = np.random.choice([-0.3, 0.3])
            t_wave = t_amplitude * np.exp(-0.5 * ((t - beat_time - 0.35) / 0.08)**2)
            signal += t_wave
        
        return signal
    
    def _generate_pvc_rhythm(self, t: np.ndarray) -> np.ndarray:
        """Generate normal rhythm with premature ventricular contractions."""
        signal = np.zeros_like(t)
        
        # Normal rhythm base
        hr = 75
        rr_base = 60.0 / hr
        current_time = 0
        beat_count = 0
        
        while current_time < t[-1]:
            beat_count += 1
            
            # Every 3rd or 4th beat is a PVC
            is_pvc = (beat_count % np.random.choice([3, 4]) == 0)
            
            if is_pvc:
                # PVC: premature, wide, different morphology
                rr_interval = rr_base * np.random.uniform(0.6, 0.8)  # Premature
                current_time += rr_interval
                
                if current_time < t[-1]:
                    # Wide QRS with abnormal morphology
                    amplitude = np.random.uniform(2.0, 3.0)
                    width_factor = np.random.uniform(1.8, 2.2)
                    
                    qrs = self._generate_qrs_complex(t, current_time, amplitude, width_factor)
                    signal += qrs
                    
                    # Compensatory pause
                    current_time += rr_base * np.random.uniform(1.2, 1.5)
            else:
                # Normal beat
                rr_interval = rr_base * np.random.normal(1.0, 0.05)
                current_time += rr_interval
                
                if current_time < t[-1]:
                    # Normal P-QRS-T complex
                    p_wave = self._generate_p_wave(t, current_time - 0.16)
                    qrs = self._generate_qrs_complex(t, current_time)
                    t_wave = self._generate_t_wave(t, current_time + 0.32)
                    
                    signal += p_wave + qrs + t_wave
        
        return signal
    
    def _generate_svt(self, t: np.ndarray) -> np.ndarray:
        """Generate supraventricular tachycardia."""
        signal = np.zeros_like(t)
        
        # Fast, regular rate (150-220 BPM)
        hr = np.random.uniform(150, 200)
        rr_interval = 60.0 / hr
        
        beat_times = np.arange(0, t[-1], rr_interval)
        
        for beat_time in beat_times:
            # Normal width QRS complexes
            qrs = self._generate_qrs_complex(t, beat_time, amplitude=0.9)
            signal += qrs
            
            # P waves may be hidden or inverted
            if np.random.random() > 0.5:
                p_amplitude = np.random.choice([-0.1, 0.1])
                p_time = beat_time + np.random.uniform(-0.1, 0.1)
                p_wave = self._generate_p_wave(t, p_time)
                p_wave *= p_amplitude / 0.15
                signal += p_wave
        
        return signal
    
    def _add_physiological_noise(self, t: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Add realistic physiological noise."""
        # Baseline wander (respiratory, movement)
        baseline_wander = (0.1 * np.sin(2 * np.pi * 0.15 * t) +  # Respiratory
                          0.05 * np.sin(2 * np.pi * 0.05 * t))    # Movement
        
        # Muscle artifacts
        muscle_noise = 0.02 * np.random.randn(len(t))
        
        # Power line interference (50/60 Hz)
        powerline = 0.01 * np.sin(2 * np.pi * 60 * t)
        
        # Electronic noise
        electronic_noise = 0.01 * np.random.randn(len(t))
        
        return baseline_wander + muscle_noise + powerline + electronic_noise
    
    def generate_dataset(self, n_normal: int = 100, n_arrhythmic: int = 100, 
                        duration: float = 10.0) -> Dict:
        """
        Generate a complete dataset for testing.
        
        Parameters:
        -----------
        n_normal : int
            Number of normal ECG signals
        n_arrhythmic : int
            Number of arrhythmic ECG signals  
        duration : float
            Duration of each signal in seconds
            
        Returns:
        --------
        dict : Dataset containing signals, labels, and metadata
        """
        signals = []
        labels = []
        metadata = []
        
        # Generate normal signals
        print(f"Generating {n_normal} normal ECG signals...")
        for i in range(n_normal):
            hr = np.random.uniform(60, 100)  # Physiological range
            t, signal = self.generate_normal_ecg(duration=duration, hr=hr)
            
            signals.append(signal)
            labels.append(0)  # Normal
            metadata.append({
                'type': 'normal',
                'heart_rate': hr,
                'duration': duration,
                'fs': self.fs
            })
        
        # Generate arrhythmic signals
        arrhythmia_types = ['atrial_fibrillation', 'ventricular_tachycardia', 
                           'premature_ventricular_contractions', 'supraventricular_tachycardia']
        
        print(f"Generating {n_arrhythmic} arrhythmic ECG signals...")
        for i in range(n_arrhythmic):
            arr_type = np.random.choice(arrhythmia_types)
            t, signal = self.generate_arrhythmic_ecg(arr_type, duration=duration)
            
            signals.append(signal)
            labels.append(1)  # Arrhythmia
            metadata.append({
                'type': arr_type,
                'duration': duration,
                'fs': self.fs
            })
        
        # Create time vector
        t = np.linspace(0, duration, int(self.fs * duration))
        
        dataset = {
            'signals': np.array(signals),
            'labels': np.array(labels),
            'metadata': metadata,
            'time': t,
            'fs': self.fs,
            'description': f'Synthetic ECG dataset with {n_normal} normal and {n_arrhythmic} arrhythmic signals'
        }
        
        print(f"Dataset generated: {len(signals)} total signals")
        print(f"  Normal: {np.sum(labels == 0)}")
        print(f"  Arrhythmic: {np.sum(labels == 1)}")
        
        return dataset


def create_sample_datasets():
    """Create and save sample datasets for testing."""
    print("Creating Sample ECG Datasets")
    print("=" * 40)
    
    generator = SyntheticECGGenerator(fs=250)
    
    # Small dataset for quick testing
    print("\n1. Creating small test dataset...")
    small_dataset = generator.generate_dataset(n_normal=20, n_arrhythmic=20, duration=5.0)
    
    # Save dataset
    data_dir = os.path.dirname(__file__)
    small_path = os.path.join(data_dir, 'small_test_dataset.pkl')
    
    with open(small_path, 'wb') as f:
        pickle.dump(small_dataset, f)
    print(f"Saved small dataset to: {small_path}")
    
    # Medium dataset for validation
    print("\n2. Creating medium validation dataset...")
    medium_dataset = generator.generate_dataset(n_normal=50, n_arrhythmic=50, duration=10.0)
    
    medium_path = os.path.join(data_dir, 'medium_validation_dataset.pkl')
    
    with open(medium_path, 'wb') as f:
        pickle.dump(medium_dataset, f)
    print(f"Saved medium dataset to: {medium_path}")
    
    # Create individual example signals
    print("\n3. Creating individual example signals...")
    examples = {}
    
    # Normal example
    t, normal_signal = generator.generate_normal_ecg(duration=10.0, hr=75)
    examples['normal'] = {'time': t, 'signal': normal_signal, 'label': 0}
    
    # Arrhythmia examples
    arrhythmia_types = ['atrial_fibrillation', 'ventricular_tachycardia', 
                       'premature_ventricular_contractions']
    
    for arr_type in arrhythmia_types:
        t, arr_signal = generator.generate_arrhythmic_ecg(arr_type, duration=10.0)
        examples[arr_type] = {'time': t, 'signal': arr_signal, 'label': 1}
    
    examples_path = os.path.join(data_dir, 'example_signals.pkl')
    
    with open(examples_path, 'wb') as f:
        pickle.dump(examples, f)
    print(f"Saved example signals to: {examples_path}")
    
    print("\n" + "=" * 40)
    print("✅ Sample datasets created successfully!")
    print("Available datasets:")
    print(f"  - {small_path}")
    print(f"  - {medium_path}")
    print(f"  - {examples_path}")


def load_sample_dataset(dataset_type: str = 'small') -> Dict:
    """
    Load a sample dataset.
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset ('small', 'medium', 'examples')
        
    Returns:
    --------
    dict : Loaded dataset
    """
    data_dir = os.path.dirname(__file__)
    
    if dataset_type == 'small':
        file_path = os.path.join(data_dir, 'small_test_dataset.pkl')
    elif dataset_type == 'medium':
        file_path = os.path.join(data_dir, 'medium_validation_dataset.pkl')
    elif dataset_type == 'examples':
        file_path = os.path.join(data_dir, 'example_signals.pkl')
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if not os.path.exists(file_path):
        print(f"Dataset {dataset_type} not found. Creating it...")
        create_sample_datasets()
    
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {dataset_type} dataset from: {file_path}")
    return dataset


if __name__ == "__main__":
    # Create sample datasets when run as script
    create_sample_datasets()