"""
Persistence Computation Module
Implements Vietoris-Rips complex construction and persistence diagram computation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings


class PersistenceComputer:
    """
    Compute persistence diagrams using various libraries (Ripser, GUDHI, Giotto-TDA).
    
    This class provides a unified interface for persistence computation,
    optimized for cardiac signal analysis.
    """
    
    def __init__(self, library: str = 'ripser', max_dimension: int = 2, 
                 max_edge_length: float = 2.0):
        """
        Initialize persistence computer.
        
        Parameters:
        -----------
        library : str
            Library to use ('ripser', 'gudhi', 'giotto')
        max_dimension : int
            Maximum homology dimension to compute (typically 2 for ECG)
        max_edge_length : float
            Maximum edge length for Vietoris-Rips complex
        """
        self.library = library.lower()
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        
        # Validate library availability
        self._check_library_availability()
        
    def _check_library_availability(self):
        """Check if requested library is available."""
        if self.library == 'ripser':
            try:
                import ripser
                self._ripser_available = True
            except ImportError:
                warnings.warn("Ripser not available. Install with: pip install ripser")
                self._ripser_available = False
                
        elif self.library == 'gudhi':
            try:
                import gudhi
                self._gudhi_available = True
            except ImportError:
                warnings.warn("GUDHI not available. Install with: pip install gudhi")
                self._gudhi_available = False
                
        elif self.library == 'giotto':
            try:
                import gtda
                self._giotto_available = True
            except ImportError:
                warnings.warn("Giotto-TDA not available. Install with: pip install giotto-tda")
                self._giotto_available = False
        else:
            raise ValueError(f"Unknown library: {self.library}")
    
    def compute_persistence(self, point_cloud: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute persistence diagrams for the given point cloud.
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Embedded signal data (n_points, n_dimensions)
            
        Returns:
        --------
        dict : Dictionary mapping dimension to persistence diagram
        """
        if self.library == 'ripser':
            return self._compute_ripser(point_cloud)
        elif self.library == 'gudhi':
            return self._compute_gudhi(point_cloud)
        elif self.library == 'giotto':
            return self._compute_giotto(point_cloud)
        else:
            raise ValueError(f"Unknown library: {self.library}")
    
    def _compute_ripser(self, point_cloud: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute persistence using Ripser (fastest option)."""
        try:
            from ripser import ripser
        except ImportError:
            raise ImportError("Ripser not installed. Install with: pip install ripser")
        
        # Compute persistence
        result = ripser(point_cloud, 
                       maxdim=self.max_dimension, 
                       thresh=self.max_edge_length)
        
        # Convert to standard format
        diagrams = {}
        for dim, dgm in enumerate(result['dgms']):
            if len(dgm) > 0:
                # Filter out infinite persistence
                finite_dgm = dgm[dgm[:, 1] < np.inf]
                diagrams[dim] = finite_dgm
            else:
                diagrams[dim] = np.array([])
        
        return diagrams
    
    def _compute_gudhi(self, point_cloud: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute persistence using GUDHI (most features)."""
        try:
            import gudhi
        except ImportError:
            raise ImportError("GUDHI not installed. Install with: pip install gudhi")
        
        # Create Rips complex
        rips_complex = gudhi.RipsComplex(
            points=point_cloud, 
            max_edge_length=self.max_edge_length
        )
        
        # Create simplex tree
        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dimension + 1
        )
        
        # Compute persistence
        persistence = simplex_tree.persistence()
        
        # Convert to standard format
        diagrams = {dim: [] for dim in range(self.max_dimension + 1)}
        
        for dim, (birth, death) in persistence:
            if death != float('inf') and dim <= self.max_dimension:
                diagrams[dim].append([birth, death])
        
        # Convert to numpy arrays
        for dim in diagrams:
            diagrams[dim] = np.array(diagrams[dim]) if diagrams[dim] else np.array([])
        
        return diagrams
    
    def _compute_giotto(self, point_cloud: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute persistence using Giotto-TDA (scikit-learn compatible)."""
        try:
            from gtda.homology import VietorisRipsPersistence
        except ImportError:
            raise ImportError("Giotto-TDA not installed. Install with: pip install giotto-tda")
        
        # Setup persistence computer
        vr = VietorisRipsPersistence(
            homology_dimensions=list(range(self.max_dimension + 1)),
            infinity_values=self.max_edge_length,
            reduced_homology=True
        )
        
        # Reshape for giotto-tda (expects 3D array)
        X = point_cloud.reshape(1, *point_cloud.shape)
        
        # Compute persistence
        persistence_diagrams = vr.fit_transform(X)[0]
        
        # Convert to standard format
        diagrams = {}
        for dim in range(self.max_dimension + 1):
            # Extract dimension-specific diagram
            dim_mask = persistence_diagrams[:, 2] == dim
            dim_diagram = persistence_diagrams[dim_mask, :2]
            
            # Filter out infinite persistence
            finite_mask = dim_diagram[:, 1] < self.max_edge_length * 0.99
            diagrams[dim] = dim_diagram[finite_mask]
        
        return diagrams
    
    def compute_betti_curves(self, diagrams: Dict[int, np.ndarray], 
                           resolution: int = 100) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute Betti curves (number of features alive at each filtration value).
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
        resolution : int
            Number of points in the curve
            
        Returns:
        --------
        dict : Dictionary mapping dimension to (filtration_values, betti_numbers)
        """
        betti_curves = {}
        
        # Find global filtration range
        all_values = []
        for dim_diagram in diagrams.values():
            if len(dim_diagram) > 0:
                all_values.extend(dim_diagram.flatten())
        
        if not all_values:
            return {}
        
        min_val, max_val = min(all_values), max(all_values)
        filtration_values = np.linspace(min_val, max_val, resolution)
        
        # Compute Betti curve for each dimension
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                betti_curves[dim] = (filtration_values, np.zeros(resolution))
                continue
            
            betti_numbers = np.zeros(resolution)
            
            for i, t in enumerate(filtration_values):
                # Count features alive at time t
                alive = np.sum((diagram[:, 0] <= t) & (diagram[:, 1] > t))
                betti_numbers[i] = alive
            
            betti_curves[dim] = (filtration_values, betti_numbers)
        
        return betti_curves
    
    def compute_persistence_statistics(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistical summaries of persistence diagrams.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
            
        Returns:
        --------
        dict : Statistical features
        """
        stats = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                # Empty diagram
                stats.update({
                    f'H{dim}_count': 0,
                    f'H{dim}_mean_persistence': 0,
                    f'H{dim}_std_persistence': 0,
                    f'H{dim}_max_persistence': 0,
                    f'H{dim}_total_persistence': 0,
                    f'H{dim}_mean_birth': 0,
                    f'H{dim}_mean_death': 0,
                    f'H{dim}_birth_spread': 0,
                    f'H{dim}_death_spread': 0
                })
                
                if dim == 0:
                    stats['latest_death_time'] = 0
                continue
            
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            persistences = deaths - births
            
            # Basic statistics
            stats[f'H{dim}_count'] = len(diagram)
            stats[f'H{dim}_mean_persistence'] = np.mean(persistences)
            stats[f'H{dim}_std_persistence'] = np.std(persistences)
            stats[f'H{dim}_max_persistence'] = np.max(persistences)
            stats[f'H{dim}_total_persistence'] = np.sum(persistences)
            
            # Birth/death statistics
            stats[f'H{dim}_mean_birth'] = np.mean(births)
            stats[f'H{dim}_mean_death'] = np.mean(deaths)
            stats[f'H{dim}_birth_spread'] = np.std(births)
            stats[f'H{dim}_death_spread'] = np.std(deaths)
            
            # Special feature for dimension 0 (latest death time)
            if dim == 0:
                stats['latest_death_time'] = np.max(deaths)
        
        return stats
    
    def compute_persistence_entropy(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Compute persistence entropy for each homology dimension.
        
        Persistence entropy measures the complexity/disorder of the persistence diagram.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
            
        Returns:
        --------
        dict : Entropy values by dimension
        """
        entropies = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                entropies[f'H{dim}_entropy'] = 0
                continue
            
            # Compute persistences
            persistences = diagram[:, 1] - diagram[:, 0]
            
            # Normalize to get probability distribution
            total_persistence = np.sum(persistences)
            
            if total_persistence > 0:
                probabilities = persistences / total_persistence
                
                # Compute entropy
                # Add small epsilon to avoid log(0)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropies[f'H{dim}_entropy'] = entropy
            else:
                entropies[f'H{dim}_entropy'] = 0
        
        return entropies
    
    def plot_persistence_diagram(self, diagrams: Dict[int, np.ndarray], 
                                dimensions: Optional[List[int]] = None,
                                ax=None, title: str = "Persistence Diagram"):
        """
        Plot persistence diagrams.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
        dimensions : list or None
            Dimensions to plot (default: all)
        ax : matplotlib axis
            Axis to plot on
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if dimensions is None:
            dimensions = list(diagrams.keys())
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'v', 'D']
        
        # Plot each dimension
        for i, dim in enumerate(dimensions):
            if dim not in diagrams or len(diagrams[dim]) == 0:
                continue
            
            diagram = diagrams[dim]
            ax.scatter(diagram[:, 0], diagram[:, 1], 
                      c=colors[i % len(colors)],
                      marker=markers[i % len(markers)],
                      s=50, alpha=0.7,
                      label=f'H{dim}')
        
        # Plot diagonal
        max_val = 0
        for diagram in diagrams.values():
            if len(diagram) > 0:
                max_val = max(max_val, np.max(diagram))
        
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_barcode(self, diagrams: Dict[int, np.ndarray], 
                    dimensions: Optional[List[int]] = None,
                    ax=None, title: str = "Persistence Barcode"):
        """
        Plot persistence barcode.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
        dimensions : list or None
            Dimensions to plot
        ax : matplotlib axis
            Axis to plot on
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if dimensions is None:
            dimensions = list(diagrams.keys())
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        # Plot bars for each dimension
        for dim_idx, dim in enumerate(dimensions):
            if dim not in diagrams or len(diagrams[dim]) == 0:
                continue
            
            diagram = diagrams[dim]
            color = colors[dim_idx % len(colors)]
            
            # Sort by birth time
            sorted_indices = np.argsort(diagram[:, 0])
            sorted_diagram = diagram[sorted_indices]
            
            for i, (birth, death) in enumerate(sorted_diagram):
                ax.barh(y_pos, death - birth, left=birth, 
                       height=0.8, color=color, alpha=0.7)
                y_pos += 1
            
            # Add dimension label
            y_ticks.append(y_pos - len(diagram)/2 - 0.5)
            y_labels.append(f'H{dim}')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Filtration Value')
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
        
        return ax