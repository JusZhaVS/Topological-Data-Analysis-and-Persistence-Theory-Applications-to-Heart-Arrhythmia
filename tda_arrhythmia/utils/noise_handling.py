"""
Noise Handling and Robustness Utilities for TDA
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d
import warnings

from ..core.embedding import TakensEmbedding
from ..core.persistence import PersistenceComputer


class TopologyPreservingDenoiser:
    """
    Denoising algorithm that preserves topological features.
    
    This denoiser iteratively removes noise while ensuring that
    significant topological features are preserved.
    """
    
    def __init__(self, persistence_threshold: float = 0.1, 
                 max_iterations: int = 10,
                 smoothing_sigma: float = 0.5):
        """
        Initialize topology-preserving denoiser.
        
        Parameters:
        -----------
        persistence_threshold : float
            Minimum relative persistence to preserve
        max_iterations : int
            Maximum denoising iterations
        smoothing_sigma : float
            Gaussian filter sigma for smoothing
        """
        self.persistence_threshold = persistence_threshold
        self.max_iterations = max_iterations
        self.smoothing_sigma = smoothing_sigma
        
        self.embedder = TakensEmbedding(dimension=3)
        self.persistence_computer = PersistenceComputer()
    
    def denoise(self, signal: np.ndarray, 
               preserve_dimensions: List[int] = [0, 1]) -> np.ndarray:
        """
        Denoise signal while preserving topology.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        preserve_dimensions : list
            Homology dimensions to preserve
            
        Returns:
        --------
        np.ndarray : Denoised signal
        """
        # Compute initial persistence
        try:
            embedded, delay = self.embedder.embed(signal)
            initial_diagrams = self.persistence_computer.compute_persistence(embedded)
        except Exception as e:
            warnings.warn(f"Failed to compute initial persistence: {e}")
            # Fallback to basic denoising
            return gaussian_filter1d(signal, sigma=self.smoothing_sigma)
        
        # Identify significant features
        significant_features = self._identify_significant_features(
            initial_diagrams, preserve_dimensions
        )
        
        # Iterative denoising
        denoised = signal.copy()
        
        for iteration in range(self.max_iterations):
            # Apply gentle smoothing
            candidate = gaussian_filter1d(
                denoised, sigma=self.smoothing_sigma
            )
            
            # Check topological preservation
            try:
                embedded_candidate, _ = self.embedder.embed(candidate, delay=delay)
                candidate_diagrams = self.persistence_computer.compute_persistence(
                    embedded_candidate
                )
                
                # Compute topological distance
                topo_distance = self._compute_topological_distance(
                    significant_features, candidate_diagrams
                )
                
                # Accept or reject candidate
                if topo_distance < 0.1:  # Acceptable distortion
                    denoised = candidate
                else:
                    # Reduce smoothing strength
                    denoised = 0.8 * candidate + 0.2 * denoised
                    
            except Exception:
                # If persistence computation fails, use conservative update
                denoised = 0.9 * denoised + 0.1 * candidate
        
        return denoised
    
    def _identify_significant_features(self, diagrams: Dict[int, np.ndarray],
                                     dimensions: List[int]) -> Dict[int, np.ndarray]:
        """Identify topologically significant features."""
        significant = {}
        
        for dim in dimensions:
            if dim not in diagrams or len(diagrams[dim]) == 0:
                significant[dim] = np.array([])
                continue
            
            diagram = diagrams[dim]
            persistences = diagram[:, 1] - diagram[:, 0]
            
            if len(persistences) > 0:
                threshold = self.persistence_threshold * np.max(persistences)
                mask = persistences > threshold
                significant[dim] = diagram[mask]
            else:
                significant[dim] = np.array([])
        
        return significant
    
    def _compute_topological_distance(self, reference: Dict[int, np.ndarray],
                                    candidate: Dict[int, np.ndarray]) -> float:
        """Compute distance between topological features."""
        total_distance = 0.0
        
        for dim in reference:
            if dim not in candidate:
                total_distance += 1.0
                continue
            
            ref_diagram = reference[dim]
            cand_diagram = candidate[dim]
            
            if len(ref_diagram) == 0 and len(cand_diagram) == 0:
                continue
            elif len(ref_diagram) == 0 or len(cand_diagram) == 0:
                total_distance += 1.0
                continue
            
            # Simple distance based on number of features and max persistence
            ref_count = len(ref_diagram)
            cand_count = len(cand_diagram)
            count_diff = abs(ref_count - cand_count) / max(ref_count, cand_count)
            
            ref_max_pers = np.max(ref_diagram[:, 1] - ref_diagram[:, 0])
            cand_max_pers = np.max(cand_diagram[:, 1] - cand_diagram[:, 0])
            pers_diff = abs(ref_max_pers - cand_max_pers) / max(ref_max_pers, cand_max_pers)
            
            total_distance += (count_diff + pers_diff) / 2
        
        return total_distance


class RobustTDAAnalyzer:
    """
    Robust TDA analysis with confidence intervals and multi-scale features.
    
    Provides statistical robustness for TDA-based cardiac analysis through
    bootstrap sampling and multi-resolution analysis.
    """
    
    def __init__(self, n_bootstrap: int = 50, confidence_level: float = 0.95):
        """
        Initialize robust TDA analyzer.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
        self.embedder = TakensEmbedding()
        self.persistence_computer = PersistenceComputer()
    
    def robust_analysis(self, signal: np.ndarray) -> Dict[str, Dict]:
        """
        Perform robust TDA analysis with confidence intervals.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input ECG signal
            
        Returns:
        --------
        dict : Features with confidence intervals
        """
        bootstrap_features = []
        
        signal_length = len(signal)
        
        for i in range(self.n_bootstrap):
            try:
                # Bootstrap sampling
                indices = np.random.choice(signal_length, signal_length, replace=True)
                bootstrap_signal = signal[indices]
                
                # Sort to maintain temporal structure approximately
                bootstrap_signal = np.sort(bootstrap_signal)
                
                # Compute TDA features
                embedded, _ = self.embedder.embed(bootstrap_signal)
                diagrams = self.persistence_computer.compute_persistence(embedded)
                features = self._extract_statistical_features(diagrams)
                
                bootstrap_features.append(features)
                
            except Exception as e:
                warnings.warn(f"Bootstrap iteration {i} failed: {e}")
                continue
        
        if not bootstrap_features:
            raise RuntimeError("All bootstrap iterations failed")
        
        # Compute confidence intervals
        return self._compute_confidence_intervals(bootstrap_features)
    
    def _extract_statistical_features(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        """Extract basic statistical features."""
        features = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                features.update({
                    f'H{dim}_count': 0,
                    f'H{dim}_mean_persistence': 0,
                    f'H{dim}_max_persistence': 0,
                    f'H{dim}_total_persistence': 0
                })
                if dim == 0:
                    features['latest_death_time'] = 0
                continue
            
            persistences = diagram[:, 1] - diagram[:, 0]
            deaths = diagram[:, 1]
            
            features[f'H{dim}_count'] = len(diagram)
            features[f'H{dim}_mean_persistence'] = np.mean(persistences)
            features[f'H{dim}_max_persistence'] = np.max(persistences)
            features[f'H{dim}_total_persistence'] = np.sum(persistences)
            
            if dim == 0:
                features['latest_death_time'] = np.max(deaths)
        
        return features
    
    def _compute_confidence_intervals(self, bootstrap_features: List[Dict]) -> Dict[str, Dict]:
        """Compute confidence intervals for features."""
        if not bootstrap_features:
            return {}
        
        feature_names = list(bootstrap_features[0].keys())
        confidence_intervals = {}
        
        alpha = 1 - self.confidence_level
        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100
        
        for feature in feature_names:
            values = [bf.get(feature, 0) for bf in bootstrap_features]
            
            confidence_intervals[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower': np.percentile(values, lower_p),
                'upper': np.percentile(values, upper_p),
                'median': np.median(values),
                'n_samples': len(values)
            }
        
        return confidence_intervals
    
    def multiscale_analysis(self, signal: np.ndarray, 
                          scales: List[float] = [0.5, 1.0, 2.0, 4.0]) -> Dict[str, float]:
        """
        Perform multi-scale TDA analysis.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        scales : list
            Scale factors for analysis
            
        Returns:
        --------
        dict : Multi-scale features
        """
        multiscale_features = {}
        
        # Analyze at each scale
        for scale in scales:
            try:
                # Apply scale-specific preprocessing
                if scale < 1.0:
                    # Upsample for fine-scale analysis
                    n_samples = int(len(signal) / scale)
                    scaled_signal = sp_signal.resample(signal, n_samples)
                else:
                    # Smooth for coarse-scale analysis
                    scaled_signal = gaussian_filter1d(signal, sigma=scale)
                
                # Compute features at this scale
                embedded, _ = self.embedder.embed(scaled_signal)
                diagrams = self.persistence_computer.compute_persistence(embedded)
                scale_features = self._extract_statistical_features(diagrams)
                
                # Add scale prefix
                for feat_name, value in scale_features.items():
                    multiscale_features[f'{feat_name}_scale_{scale}'] = value
                    
            except Exception as e:
                warnings.warn(f"Scale {scale} analysis failed: {e}")
                continue
        
        # Compute cross-scale statistics
        cross_scale_features = self._compute_cross_scale_features(
            multiscale_features, scales
        )
        multiscale_features.update(cross_scale_features)
        
        return multiscale_features
    
    def _compute_cross_scale_features(self, multiscale_features: Dict[str, float],
                                    scales: List[float]) -> Dict[str, float]:
        """Compute features that capture cross-scale relationships."""
        cross_scale = {}
        
        # Group features by base name
        base_features = set()
        for feat_name in multiscale_features:
            if '_scale_' in feat_name:
                base_name = feat_name.split('_scale_')[0]
                base_features.add(base_name)
        
        # Compute cross-scale statistics for each base feature
        for base_feat in base_features:
            values = []
            valid_scales = []
            
            for scale in scales:
                feat_name = f'{base_feat}_scale_{scale}'
                if feat_name in multiscale_features:
                    values.append(multiscale_features[feat_name])
                    valid_scales.append(scale)
            
            if len(values) > 1:
                values = np.array(values)
                valid_scales = np.array(valid_scales)
                
                # Cross-scale statistics
                cross_scale[f'{base_feat}_scale_mean'] = np.mean(values)
                cross_scale[f'{base_feat}_scale_std'] = np.std(values)
                
                # Linear trend across scales
                if len(values) > 2:
                    try:
                        trend_coeff = np.polyfit(valid_scales, values, 1)[0]
                        cross_scale[f'{base_feat}_scale_trend'] = trend_coeff
                    except:
                        cross_scale[f'{base_feat}_scale_trend'] = 0
                
                # Scale stability (coefficient of variation)
                if np.mean(values) != 0:
                    cv = np.std(values) / np.abs(np.mean(values))
                    cross_scale[f'{base_feat}_scale_stability'] = 1 / (1 + cv)
                else:
                    cross_scale[f'{base_feat}_scale_stability'] = 1
        
        return cross_scale
    
    def noise_sensitivity_analysis(self, signal: np.ndarray,
                                  noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
        """
        Analyze sensitivity to additive noise.
        
        Parameters:
        -----------
        signal : np.ndarray
            Clean signal
        noise_levels : list
            Noise standard deviations relative to signal std
            
        Returns:
        --------
        dict : Noise sensitivity analysis
        """
        signal_std = np.std(signal)
        
        # Baseline features (clean signal)
        try:
            embedded, _ = self.embedder.embed(signal)
            diagrams = self.persistence_computer.compute_persistence(embedded)
            baseline_features = self._extract_statistical_features(diagrams)
        except Exception as e:
            raise RuntimeError(f"Failed to analyze clean signal: {e}")
        
        sensitivity_results = {
            'baseline': baseline_features,
            'noise_analysis': {}
        }
        
        for noise_level in noise_levels:
            noise_results = []
            
            # Multiple realizations for each noise level
            for _ in range(20):
                try:
                    # Add noise
                    noise = np.random.normal(0, noise_level * signal_std, len(signal))
                    noisy_signal = signal + noise
                    
                    # Analyze noisy signal
                    embedded, _ = self.embedder.embed(noisy_signal)
                    diagrams = self.persistence_computer.compute_persistence(embedded)
                    noisy_features = self._extract_statistical_features(diagrams)
                    
                    noise_results.append(noisy_features)
                    
                except Exception:
                    continue
            
            if noise_results:
                # Compute statistics across realizations
                sensitivity_results['noise_analysis'][noise_level] = \
                    self._compute_confidence_intervals(noise_results)
        
        # Compute sensitivity metrics
        sensitivity_metrics = self._compute_sensitivity_metrics(
            baseline_features, sensitivity_results['noise_analysis']
        )
        sensitivity_results['sensitivity_metrics'] = sensitivity_metrics
        
        return sensitivity_results
    
    def _compute_sensitivity_metrics(self, baseline: Dict[str, float],
                                   noise_analysis: Dict) -> Dict[str, List[float]]:
        """Compute sensitivity metrics for each feature."""
        sensitivity_metrics = {}
        
        for feature_name in baseline:
            baseline_value = baseline[feature_name]
            sensitivities = []
            
            for noise_level, stats in noise_analysis.items():
                if feature_name in stats:
                    noisy_mean = stats[feature_name]['mean']
                    
                    # Relative change
                    if baseline_value != 0:
                        rel_change = abs(noisy_mean - baseline_value) / abs(baseline_value)
                    else:
                        rel_change = abs(noisy_mean)
                    
                    sensitivities.append(rel_change)
                else:
                    sensitivities.append(float('inf'))
            
            sensitivity_metrics[feature_name] = sensitivities
        
        return sensitivity_metrics