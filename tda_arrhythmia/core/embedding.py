"""
Takens' Embedding Implementation with Mutual Information
"""

import numpy as np
from scipy import signal as sp_signal
from sklearn.feature_selection import mutual_info_regression
from typing import Tuple, Optional, Union


class TakensEmbedding:
    """
    Takens' Embedding with optimal time delay selection using mutual information.
    
    Implements the method described in the technical guide for phase space reconstruction
    of time series data, particularly suited for ECG signals.
    """
    
    def __init__(self, dimension: int = 3, delay: Optional[int] = None, 
                 max_lag: int = 100, method: str = 'mutual_information'):
        """
        Initialize Takens' Embedding.
        
        Parameters:
        -----------
        dimension : int
            Embedding dimension (typically 2-3 for ECG)
        delay : int or None
            Time delay. If None, will be computed automatically
        max_lag : int
            Maximum lag to consider for delay selection
        method : str
            Method for delay selection ('mutual_information' or 'autocorrelation')
        """
        self.dimension = dimension
        self.delay = delay
        self.max_lag = max_lag
        self.method = method
        self._optimal_delay = None
        
    def find_optimal_delay(self, signal: np.ndarray) -> int:
        """
        Find optimal time delay using mutual information.
        
        The optimal delay is found at the first minimum of the mutual information
        function I(τ) = ΣΣ P(h,k)(τ) log[P(h,k)(τ)/(P(h) × P(k))]
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series signal
            
        Returns:
        --------
        int : Optimal time delay
        """
        if self.method == 'mutual_information':
            return self._find_delay_mutual_info(signal)
        elif self.method == 'autocorrelation':
            return self._find_delay_autocorr(signal)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _find_delay_mutual_info(self, signal: np.ndarray) -> int:
        """Find delay using mutual information method."""
        mi_values = []
        
        # Compute mutual information for different lags
        for lag in range(1, min(self.max_lag, len(signal) // 4)):
            # Create lagged versions
            x = signal[:-lag].reshape(-1, 1)
            y = signal[lag:]
            
            # Compute mutual information
            mi = mutual_info_regression(x, y, random_state=42)[0]
            mi_values.append(mi)
        
        # Find first minimum
        mi_values = np.array(mi_values)
        
        # Look for first local minimum
        for i in range(1, len(mi_values) - 1):
            if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                return i + 1
        
        # If no local minimum found, use global minimum
        return np.argmin(mi_values) + 1
    
    def _find_delay_autocorr(self, signal: np.ndarray) -> int:
        """Find delay using autocorrelation method (first zero crossing or 1/e decay)."""
        # Compute autocorrelation
        autocorr = np.correlate(signal - np.mean(signal), 
                               signal - np.mean(signal), 
                               mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find first zero crossing or 1/e decay
        threshold = 1.0 / np.e
        
        for i in range(1, min(len(autocorr), self.max_lag)):
            if autocorr[i] < threshold:
                return i
        
        return self.max_lag // 2
    
    def embed(self, signal: np.ndarray, delay: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Perform Takens' embedding on the signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series signal
        delay : int or None
            Time delay to use. If None, will compute optimal delay
            
        Returns:
        --------
        embedded : np.ndarray
            Embedded signal with shape (N - (m-1)*τ, m)
        delay : int
            Time delay used
        """
        # Determine delay
        if delay is None:
            if self.delay is None:
                delay = self.find_optimal_delay(signal)
                self._optimal_delay = delay
            else:
                delay = self.delay
        
        # Validate parameters
        N = len(signal)
        min_length = (self.dimension - 1) * delay + 1
        if N < min_length:
            raise ValueError(
                f"Signal too short. Need at least {min_length} samples, "
                f"but got {N}"
            )
        
        # Perform embedding
        M = N - (self.dimension - 1) * delay
        embedded = np.zeros((M, self.dimension))
        
        for i in range(self.dimension):
            embedded[:, i] = signal[i*delay:N-(self.dimension-1-i)*delay]
        
        return embedded, delay
    
    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Fit and transform the signal (sklearn-compatible interface).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series signal
            
        Returns:
        --------
        np.ndarray : Embedded signal
        """
        embedded, _ = self.embed(signal)
        return embedded
    
    def get_optimal_parameters(self, signal: np.ndarray, 
                             dimension_range: Tuple[int, int] = (2, 5)) -> dict:
        """
        Find optimal embedding parameters using False Nearest Neighbors.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        dimension_range : tuple
            Range of dimensions to test
            
        Returns:
        --------
        dict : Optimal parameters
        """
        # Find optimal delay first
        optimal_delay = self.find_optimal_delay(signal)
        
        # Test different dimensions using FNN
        fnn_percentages = []
        
        for dim in range(dimension_range[0], dimension_range[1] + 1):
            self.dimension = dim
            embedded, _ = self.embed(signal, delay=optimal_delay)
            
            # Compute FNN percentage
            fnn = self._compute_fnn(embedded)
            fnn_percentages.append(fnn)
        
        # Find where FNN drops below threshold
        threshold = 0.05  # 5% false nearest neighbors
        optimal_dim = dimension_range[0]
        
        for i, fnn in enumerate(fnn_percentages):
            if fnn < threshold:
                optimal_dim = dimension_range[0] + i
                break
        
        return {
            'dimension': optimal_dim,
            'delay': optimal_delay,
            'fnn_percentages': fnn_percentages
        }
    
    def _compute_fnn(self, embedded: np.ndarray, rtol: float = 15.0) -> float:
        """
        Compute False Nearest Neighbors percentage.
        
        Parameters:
        -----------
        embedded : np.ndarray
            Embedded signal
        rtol : float
            Tolerance ratio for FNN test
            
        Returns:
        --------
        float : Percentage of false nearest neighbors
        """
        n_points = len(embedded)
        if n_points < 10:
            return 1.0
        
        # Build distance matrix for current dimension
        from sklearn.neighbors import NearestNeighbors
        
        # Current dimension
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
        nbrs.fit(embedded[:-1])  # Exclude last point
        distances, indices = nbrs.kneighbors(embedded[:-1])
        
        # Get nearest neighbor distances (second column, first is self)
        nn_distances = distances[:, 1]
        nn_indices = indices[:, 1]
        
        # Check in next dimension
        false_count = 0
        valid_count = 0
        
        for i in range(len(nn_distances)):
            if nn_distances[i] > 0:  # Avoid division by zero
                # Distance in next dimension
                next_dim_dist = np.abs(embedded[i+1, -1] - embedded[nn_indices[i]+1, -1])
                
                # Check FNN criterion
                if next_dim_dist / nn_distances[i] > rtol:
                    false_count += 1
                valid_count += 1
        
        if valid_count == 0:
            return 1.0
        
        return false_count / valid_count
    
    def plot_mutual_information(self, signal: np.ndarray, ax=None):
        """
        Plot mutual information vs lag for visualization.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        ax : matplotlib axis
            Axis to plot on
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute MI for range of lags
        lags = range(1, min(self.max_lag, len(signal) // 4))
        mi_values = []
        
        for lag in lags:
            x = signal[:-lag].reshape(-1, 1)
            y = signal[lag:]
            mi = mutual_info_regression(x, y, random_state=42)[0]
            mi_values.append(mi)
        
        # Plot
        ax.plot(lags, mi_values, 'b-', linewidth=2)
        
        # Mark optimal delay
        optimal_delay = self.find_optimal_delay(signal)
        ax.axvline(x=optimal_delay, color='r', linestyle='--', 
                  label=f'Optimal delay = {optimal_delay}')
        
        ax.set_xlabel('Lag (samples)')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Mutual Information vs Time Delay')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax