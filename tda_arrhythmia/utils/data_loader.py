"""
PhysioNet Database Loading Utilities
Implements data loading for various cardiac databases
"""

import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Union
import os


class PhysioNetLoader:
    """
    Load and preprocess data from PhysioNet databases.
    
    Supports:
    - MIT-BIH Arrhythmia Database
    - CU Ventricular Tachyarrhythmia Database
    - CEBSDB (Combined ECG-Breathing-Seismocardiogram Database)
    - PhysioNet Computing in Cardiology Challenge datasets
    """
    
    def __init__(self, database: str, data_dir: Optional[str] = None):
        """
        Initialize PhysioNet loader.
        
        Parameters:
        -----------
        database : str
            Database name ('mitdb', 'cudb', 'cebsdb', etc.)
        data_dir : str
            Local directory containing database files
        """
        self.database = database.lower()
        self.data_dir = data_dir
        
        # Check if wfdb is available
        try:
            import wfdb
            self.wfdb = wfdb
            self._wfdb_available = True
        except ImportError:
            warnings.warn("WFDB package not installed. Install with: pip install wfdb")
            self._wfdb_available = False
    
    def load_records(self, record_names: Optional[List[str]] = None,
                    download: bool = True) -> List[Dict]:
        """
        Load records from PhysioNet database.
        
        Parameters:
        -----------
        record_names : list
            Specific records to load (None for all)
        download : bool
            Whether to download from PhysioNet if not local
            
        Returns:
        --------
        list : List of record dictionaries
        """
        if not self._wfdb_available:
            raise ImportError("WFDB not installed. Install with: pip install wfdb")
        
        if self.database == 'mitdb':
            return self._load_mitdb(record_names, download)
        elif self.database == 'cudb':
            return self._load_cudb(record_names, download)
        elif self.database == 'cebsdb':
            return self._load_cebsdb(record_names, download)
        else:
            raise ValueError(f"Unknown database: {self.database}")
    
    def _load_mitdb(self, record_names: Optional[List[str]] = None,
                   download: bool = True) -> List[Dict]:
        """Load MIT-BIH Arrhythmia Database."""
        if record_names is None:
            # Standard MIT-BIH record names
            record_names = [
                '100', '101', '102', '103', '104', '105', '106', '107',
                '108', '109', '111', '112', '113', '114', '115', '116',
                '117', '118', '119', '121', '122', '123', '124', '200',
                '201', '202', '203', '205', '207', '208', '209', '210',
                '212', '213', '214', '215', '217', '219', '220', '221',
                '222', '223', '228', '230', '231', '232', '233', '234'
            ]
        
        records = []
        pn_dir = 'mitdb/1.0.0/' if download else self.data_dir
        
        for record_name in record_names:
            try:
                # Load signal
                if download:
                    record = self.wfdb.rdrecord(record_name, pn_dir=pn_dir)
                else:
                    record = self.wfdb.rdrecord(os.path.join(self.data_dir, record_name))
                
                # Load annotations
                if download:
                    ann = self.wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
                else:
                    ann = self.wfdb.rdann(os.path.join(self.data_dir, record_name), 'atr')
                
                # Extract segments and labels
                segments = self._extract_segments_mitdb(record, ann)
                
                records.append({
                    'record_name': record_name,
                    'signal': record.p_signal,
                    'fs': record.fs,
                    'annotations': ann,
                    'segments': segments,
                    'n_channels': record.n_sig,
                    'sig_names': record.sig_name,
                    'units': record.units
                })
                
            except Exception as e:
                warnings.warn(f"Failed to load record {record_name}: {e}")
                continue
        
        return records
    
    def _extract_segments_mitdb(self, record, ann) -> List[Dict]:
        """Extract labeled segments from MIT-BIH record."""
        segments = []
        
        # Map annotation symbols to arrhythmia types
        arrhythmia_symbols = {
            'V': 'PVC',  # Premature ventricular contraction
            'F': 'Fusion',  # Fusion of ventricular and normal beat
            'S': 'SVEB',  # Supraventricular ectopic beat
            'Q': 'Unclassifiable',  # Unclassifiable beat
            '[': 'VF_start',  # Start of ventricular fibrillation
            '!': 'VF',  # Ventricular flutter
            ']': 'VF_end',  # End of ventricular fibrillation
            'A': 'APB',  # Atrial premature beat
            'a': 'Aberrated_APB',  # Aberrated atrial premature beat
            'J': 'Junction',  # Nodal (junctional) premature beat
            'E': 'VE',  # Ventricular escape beat
        }
        
        # Extract 5-second segments around each annotation
        window_size = 5 * record.fs  # 5 seconds
        half_window = window_size // 2
        
        for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            # Skip normal beats and non-beat annotations
            if symbol == 'N' or symbol in ['+', '~', '|', 'x']:
                continue
            
            # Get segment boundaries
            start = max(0, sample - half_window)
            end = min(len(record.p_signal), sample + half_window)
            
            if end - start < window_size // 2:  # Too short
                continue
            
            segment = {
                'start': start,
                'end': end,
                'center': sample,
                'symbol': symbol,
                'arrhythmia_type': arrhythmia_symbols.get(symbol, 'Other'),
                'signal': record.p_signal[start:end, 0]  # First channel
            }
            
            segments.append(segment)
        
        return segments
    
    def _load_cudb(self, record_names: Optional[List[str]] = None,
                  download: bool = True) -> List[Dict]:
        """Load CU Ventricular Tachyarrhythmia Database."""
        if record_names is None:
            # CU database has 35 records
            record_names = [f'cu{i:02d}' for i in range(1, 36)]
        
        records = []
        pn_dir = 'cudb/1.0.0/' if download else self.data_dir
        
        for record_name in record_names:
            try:
                # Load signal
                if download:
                    record = self.wfdb.rdrecord(record_name, pn_dir=pn_dir)
                else:
                    record = self.wfdb.rdrecord(os.path.join(self.data_dir, record_name))
                
                # Load annotations
                if download:
                    ann = self.wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
                else:
                    ann = self.wfdb.rdann(os.path.join(self.data_dir, record_name), 'atr')
                
                # Find VF/VT episodes
                vf_episodes = self._extract_vf_episodes(ann)
                
                records.append({
                    'record_name': record_name,
                    'signal': record.p_signal[:, 0],  # First channel
                    'fs': record.fs,
                    'annotations': ann,
                    'vf_episodes': vf_episodes,
                    'has_vf': len(vf_episodes) > 0,
                    'duration': len(record.p_signal) / record.fs
                })
                
            except Exception as e:
                warnings.warn(f"Failed to load record {record_name}: {e}")
                continue
        
        return records
    
    def _extract_vf_episodes(self, ann) -> List[Dict]:
        """Extract VF/VT episodes from annotations."""
        episodes = []
        vf_start = None
        
        for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            if symbol == '[':  # VF/VT start
                vf_start = sample
            elif symbol == ']' and vf_start is not None:  # VF/VT end
                episodes.append({
                    'start': vf_start,
                    'end': sample,
                    'duration': sample - vf_start,
                    'type': 'VF/VT'
                })
                vf_start = None
        
        return episodes
    
    def _load_cebsdb(self, record_names: Optional[List[str]] = None,
                    download: bool = True) -> List[Dict]:
        """Load Combined ECG-Breathing-Seismocardiogram Database."""
        if record_names is None:
            # CEBSDB has 20 subjects
            record_names = [f'b{i:03d}r' for i in range(1, 21)]
        
        records = []
        pn_dir = 'cebsdb/1.0.0/' if download else self.data_dir
        
        for record_name in record_names:
            try:
                # Load multi-modal signals
                if download:
                    record = self.wfdb.rdrecord(record_name, pn_dir=pn_dir)
                else:
                    record = self.wfdb.rdrecord(os.path.join(self.data_dir, record_name))
                
                # Extract different modalities
                ecg_i = record.p_signal[:, 0]      # ECG Lead I
                ecg_ii = record.p_signal[:, 1]     # ECG Lead II  
                ecg_iii = record.p_signal[:, 2]    # ECG Lead III
                respiration = record.p_signal[:, 3] # Respiratory signal
                scg = record.p_signal[:, 4]        # Seismocardiogram
                
                # Segment by experimental phases (5kHz sampling)
                fs = record.fs
                phases = {
                    'baseline': (0, 5*60*fs),           # First 5 minutes
                    'music': (5*60*fs, 55*60*fs),       # Next 50 minutes
                    'recovery': (55*60*fs, 60*60*fs)    # Last 5 minutes
                }
                
                records.append({
                    'record_name': record_name,
                    'ecg_i': ecg_i,
                    'ecg_ii': ecg_ii,
                    'ecg_iii': ecg_iii,
                    'respiration': respiration,
                    'scg': scg,
                    'fs': fs,
                    'phases': phases,
                    'duration': len(record.p_signal) / fs
                })
                
            except Exception as e:
                warnings.warn(f"Failed to load record {record_name}: {e}")
                continue
        
        return records
    
    def create_dataset(self, records: List[Dict], 
                      segment_length: float = 5.0,
                      overlap: float = 0.5,
                      balance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dataset from loaded records.
        
        Parameters:
        -----------
        records : list
            Loaded records
        segment_length : float
            Length of segments in seconds
        overlap : float
            Overlap fraction between segments
        balance : bool
            Whether to balance classes
            
        Returns:
        --------
        X : np.ndarray
            Signal segments
        y : np.ndarray
            Binary labels (0: normal, 1: arrhythmia)
        """
        segments = []
        labels = []
        
        for record in records:
            if self.database == 'mitdb':
                # Use pre-extracted segments
                for seg in record.get('segments', []):
                    segments.append(seg['signal'])
                    labels.append(1)  # Arrhythmia
                
                # Also extract normal segments
                normal_segments = self._extract_normal_segments(
                    record, segment_length
                )
                segments.extend(normal_segments)
                labels.extend([0] * len(normal_segments))
                
            elif self.database == 'cudb':
                # Extract segments with/without VF
                vf_segments, normal_segments = self._extract_cudb_segments(
                    record, segment_length, overlap
                )
                segments.extend(vf_segments)
                labels.extend([1] * len(vf_segments))
                segments.extend(normal_segments)
                labels.extend([0] * len(normal_segments))
        
        # Convert to arrays
        X = np.array(segments)
        y = np.array(labels)
        
        # Balance classes if requested
        if balance:
            X, y = self._balance_dataset(X, y)
        
        return X, y
    
    def _extract_normal_segments(self, record: Dict, 
                               segment_length: float) -> List[np.ndarray]:
        """Extract normal (non-arrhythmic) segments."""
        segments = []
        fs = record['fs']
        segment_samples = int(segment_length * fs)
        signal = record['signal'][:, 0]  # First channel
        
        # Get arrhythmia regions to avoid
        arrhythmia_regions = []
        for seg in record.get('segments', []):
            arrhythmia_regions.append((seg['start'], seg['end']))
        
        # Extract non-overlapping normal segments
        start = 0
        while start + segment_samples < len(signal):
            end = start + segment_samples
            
            # Check if segment overlaps with any arrhythmia
            is_normal = True
            for arr_start, arr_end in arrhythmia_regions:
                if not (end < arr_start or start > arr_end):
                    is_normal = False
                    break
            
            if is_normal:
                segments.append(signal[start:end])
            
            start += segment_samples // 2  # 50% overlap
        
        return segments
    
    def _extract_cudb_segments(self, record: Dict, segment_length: float,
                             overlap: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract VF and normal segments from CU database."""
        vf_segments = []
        normal_segments = []
        
        signal = record['signal']
        fs = record['fs']
        segment_samples = int(segment_length * fs)
        step = int(segment_samples * (1 - overlap))
        
        # Extract segments around VF episodes
        for episode in record['vf_episodes']:
            # Pre-VF segment (1 minute before)
            pre_start = max(0, episode['start'] - 60*fs)
            pre_end = episode['start']
            
            if pre_end - pre_start >= segment_samples:
                vf_segments.append(signal[pre_end-segment_samples:pre_end])
            
            # During VF segments
            vf_start = episode['start']
            vf_end = episode['end']
            
            pos = vf_start
            while pos + segment_samples <= vf_end:
                vf_segments.append(signal[pos:pos+segment_samples])
                pos += step
        
        # Extract normal segments (far from VF)
        safe_distance = 5 * 60 * fs  # 5 minutes
        
        pos = 0
        while pos + segment_samples < len(signal):
            # Check distance to nearest VF
            is_safe = True
            for episode in record['vf_episodes']:
                if abs(pos - episode['start']) < safe_distance or \
                   abs(pos - episode['end']) < safe_distance:
                    is_safe = False
                    break
            
            if is_safe:
                normal_segments.append(signal[pos:pos+segment_samples])
            
            pos += step
        
        return vf_segments, normal_segments
    
    def _balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset by undersampling majority class."""
        # Count classes
        n_class_0 = np.sum(y == 0)
        n_class_1 = np.sum(y == 1)
        
        if n_class_0 == n_class_1:
            return X, y
        
        # Undersample majority class
        if n_class_0 > n_class_1:
            # Undersample class 0
            indices_0 = np.where(y == 0)[0]
            indices_1 = np.where(y == 1)[0]
            
            # Random sample from majority class
            selected_0 = np.random.choice(indices_0, n_class_1, replace=False)
            selected_indices = np.concatenate([selected_0, indices_1])
        else:
            # Undersample class 1
            indices_0 = np.where(y == 0)[0]
            indices_1 = np.where(y == 1)[0]
            
            selected_1 = np.random.choice(indices_1, n_class_0, replace=False)
            selected_indices = np.concatenate([indices_0, selected_1])
        
        # Shuffle
        np.random.shuffle(selected_indices)
        
        return X[selected_indices], y[selected_indices]
    
    def get_record_info(self, record_name: str) -> Dict:
        """Get detailed information about a specific record."""
        records = self.load_records([record_name])
        
        if not records:
            raise ValueError(f"Record {record_name} not found")
        
        record = records[0]
        info = {
            'name': record_name,
            'duration': record.get('duration', 0),
            'sampling_frequency': record.get('fs', 0),
            'n_channels': record.get('n_channels', 0),
            'signal_names': record.get('sig_names', []),
            'units': record.get('units', [])
        }
        
        if self.database == 'mitdb':
            info['n_arrhythmia_segments'] = len(record.get('segments', []))
            info['arrhythmia_types'] = list(set(
                seg['arrhythmia_type'] for seg in record.get('segments', [])
            ))
        elif self.database == 'cudb':
            info['has_vf'] = record.get('has_vf', False)
            info['n_vf_episodes'] = len(record.get('vf_episodes', []))
            if info['n_vf_episodes'] > 0:
                total_vf_duration = sum(
                    ep['duration'] for ep in record['vf_episodes']
                ) / record['fs']
                info['total_vf_duration'] = total_vf_duration
        
        return info