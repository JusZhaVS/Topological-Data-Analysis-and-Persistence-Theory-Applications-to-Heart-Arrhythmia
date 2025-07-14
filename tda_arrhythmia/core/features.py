"""
Persistence Feature Extraction Module
Implements comprehensive feature extraction from persistence diagrams
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import entropy
from scipy.spatial.distance import cdist


class PersistenceFeatureExtractor:
    """
    Extract features from persistence diagrams for machine learning.
    
    Implements various feature extraction methods including:
    - Statistical features
    - Persistence landscapes
    - Persistence images
    - Topological triangles
    - Multi-scale features
    """
    
    def __init__(self, feature_types: List[str] = None):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        feature_types : list
            List of feature types to extract. Options:
            'statistics', 'landscapes', 'images', 'triangles', 'entropy'
        """
        if feature_types is None:
            feature_types = ['statistics', 'entropy']
        
        self.feature_types = feature_types
        self._validate_feature_types()
    
    def _validate_feature_types(self):
        """Validate requested feature types."""
        valid_types = {'statistics', 'landscapes', 'images', 'triangles', 
                      'entropy', 'kernels', 'vectors'}
        
        for ftype in self.feature_types:
            if ftype not in valid_types:
                raise ValueError(f"Unknown feature type: {ftype}")
    
    def extract_features(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract all requested features from persistence diagrams.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
            
        Returns:
        --------
        dict : Extracted features
        """
        features = {}
        
        if 'statistics' in self.feature_types:
            features.update(self.extract_statistical_features(diagrams))
        
        if 'entropy' in self.feature_types:
            features.update(self.extract_entropy_features(diagrams))
        
        if 'landscapes' in self.feature_types:
            features.update(self.extract_landscape_features(diagrams))
        
        if 'images' in self.feature_types:
            features.update(self.extract_persistence_images(diagrams))
        
        if 'triangles' in self.feature_types:
            features.update(self.extract_topological_triangles(diagrams))
        
        if 'vectors' in self.feature_types:
            features.update(self.extract_persistence_vectors(diagrams))
        
        return features
    
    def extract_statistical_features(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Extract comprehensive statistical features from persistence diagrams.
        
        Includes the key "latest death time" feature for 0-dimensional homology.
        """
        features = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                # Empty diagram - fill with zeros
                feature_names = [
                    f'H{dim}_count', f'H{dim}_mean_persistence', 
                    f'H{dim}_std_persistence', f'H{dim}_max_persistence',
                    f'H{dim}_total_persistence', f'H{dim}_mean_birth',
                    f'H{dim}_mean_death', f'H{dim}_birth_spread',
                    f'H{dim}_death_spread', f'H{dim}_midlife_mean',
                    f'H{dim}_midlife_std'
                ]
                features.update({name: 0.0 for name in feature_names})
                
                if dim == 0:
                    features['latest_death_time'] = 0.0
                    features['H0_second_latest_death'] = 0.0
                continue
            
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            persistences = deaths - births
            midlifes = (births + deaths) / 2
            
            # Basic statistics
            features[f'H{dim}_count'] = float(len(diagram))
            features[f'H{dim}_mean_persistence'] = float(np.mean(persistences))
            features[f'H{dim}_std_persistence'] = float(np.std(persistences))
            features[f'H{dim}_max_persistence'] = float(np.max(persistences))
            features[f'H{dim}_total_persistence'] = float(np.sum(persistences))
            
            # Birth/death statistics
            features[f'H{dim}_mean_birth'] = float(np.mean(births))
            features[f'H{dim}_mean_death'] = float(np.mean(deaths))
            features[f'H{dim}_birth_spread'] = float(np.std(births))
            features[f'H{dim}_death_spread'] = float(np.std(deaths))
            
            # Midlife statistics
            features[f'H{dim}_midlife_mean'] = float(np.mean(midlifes))
            features[f'H{dim}_midlife_std'] = float(np.std(midlifes))
            
            # Percentile features
            for p in [25, 50, 75]:
                features[f'H{dim}_persistence_p{p}'] = float(np.percentile(persistences, p))
            
            # Special features for dimension 0
            if dim == 0:
                # Latest death time - key feature from the paper
                features['latest_death_time'] = float(np.max(deaths))
                
                # Second latest death (for robustness)
                if len(deaths) > 1:
                    sorted_deaths = np.sort(deaths)
                    features['H0_second_latest_death'] = float(sorted_deaths[-2])
                else:
                    features['H0_second_latest_death'] = 0.0
                
                # Gap between latest deaths
                if len(deaths) > 1:
                    sorted_deaths = np.sort(deaths)
                    features['H0_death_gap'] = float(sorted_deaths[-1] - sorted_deaths[-2])
                else:
                    features['H0_death_gap'] = 0.0
            
            # Features for dimension 1 (loops)
            if dim == 1 and len(diagram) > 0:
                # Find most persistent loops
                sorted_indices = np.argsort(persistences)[::-1]
                top_k = min(3, len(diagram))
                
                for i in range(top_k):
                    idx = sorted_indices[i]
                    features[f'H1_loop{i+1}_birth'] = float(births[idx])
                    features[f'H1_loop{i+1}_persistence'] = float(persistences[idx])
        
        return features
    
    def extract_entropy_features(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Extract entropy-based features from persistence diagrams.
        """
        features = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                features[f'H{dim}_entropy'] = 0.0
                features[f'H{dim}_normalized_entropy'] = 0.0
                continue
            
            persistences = diagram[:, 1] - diagram[:, 0]
            
            # Persistence entropy
            if np.sum(persistences) > 0:
                # Normalize to probability distribution
                p = persistences / np.sum(persistences)
                # Compute entropy
                h = -np.sum(p * np.log2(p + 1e-10))
                features[f'H{dim}_entropy'] = float(h)
                
                # Normalized entropy (0 to 1)
                max_entropy = np.log2(len(persistences))
                if max_entropy > 0:
                    features[f'H{dim}_normalized_entropy'] = float(h / max_entropy)
                else:
                    features[f'H{dim}_normalized_entropy'] = 0.0
            else:
                features[f'H{dim}_entropy'] = 0.0
                features[f'H{dim}_normalized_entropy'] = 0.0
            
            # Life entropy (using lifespans)
            if len(diagram) > 1:
                births = diagram[:, 0]
                life_entropy = entropy(births + 1e-10)  # Add epsilon to avoid log(0)
                features[f'H{dim}_life_entropy'] = float(life_entropy)
            else:
                features[f'H{dim}_life_entropy'] = 0.0
        
        return features
    
    def extract_landscape_features(self, diagrams: Dict[int, np.ndarray], 
                                  n_layers: int = 5, n_bins: int = 100) -> Dict[str, np.ndarray]:
        """
        Extract persistence landscape features.
        
        Persistence landscapes provide a functional representation of persistence diagrams.
        """
        try:
            from gtda.diagrams import PersistenceLandscape
            use_gtda = True
        except ImportError:
            use_gtda = False
        
        features = {}
        
        if use_gtda:
            # Use Giotto-TDA implementation
            pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
            
            # Convert diagrams to Giotto format
            all_points = []
            for dim, diagram in diagrams.items():
                if len(diagram) > 0:
                    # Add dimension column
                    dim_col = np.full((len(diagram), 1), dim)
                    points = np.hstack([diagram, dim_col])
                    all_points.append(points)
            
            if all_points:
                combined_diagram = np.vstack(all_points).reshape(1, -1, 3)
                landscapes = pl.fit_transform(combined_diagram)
                features['persistence_landscapes'] = landscapes.flatten()
            else:
                features['persistence_landscapes'] = np.zeros(n_layers * n_bins)
        else:
            # Manual implementation
            for dim, diagram in diagrams.items():
                if len(diagram) == 0:
                    features[f'H{dim}_landscape'] = np.zeros(n_layers * n_bins)
                    continue
                
                landscape = self._compute_landscape_manual(diagram, n_layers, n_bins)
                features[f'H{dim}_landscape'] = landscape.flatten()
        
        return features
    
    def _compute_landscape_manual(self, diagram: np.ndarray, 
                                 n_layers: int, n_bins: int) -> np.ndarray:
        """
        Manually compute persistence landscape.
        """
        if len(diagram) == 0:
            return np.zeros((n_layers, n_bins))
        
        # Define grid
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        min_val = np.min(births)
        max_val = np.max(deaths)
        
        grid = np.linspace(min_val, max_val, n_bins)
        landscapes = np.zeros((n_layers, n_bins))
        
        # For each grid point
        for i, t in enumerate(grid):
            # Compute landscape functions for all points
            values = []
            
            for b, d in diagram:
                if b <= t <= d:
                    # Point contributes to landscape
                    val = min(t - b, d - t)
                    values.append(val)
            
            # Sort values in descending order
            values = sorted(values, reverse=True)
            
            # Assign to layers
            for layer in range(min(n_layers, len(values))):
                landscapes[layer, i] = values[layer]
        
        return landscapes
    
    def extract_persistence_images(self, diagrams: Dict[int, np.ndarray],
                                  sigma: float = 0.1, n_bins: int = 20) -> Dict[str, np.ndarray]:
        """
        Extract persistence images for CNN processing.
        """
        try:
            from gtda.diagrams import PersistenceImage
            use_gtda = True
        except ImportError:
            use_gtda = False
        
        features = {}
        
        if use_gtda:
            # Use Giotto-TDA implementation
            pi = PersistenceImage(sigma=sigma, n_bins=n_bins)
            
            for dim, diagram in diagrams.items():
                if len(diagram) == 0:
                    features[f'H{dim}_image'] = np.zeros((n_bins, n_bins))
                    continue
                
                # Add dimension column
                dim_col = np.full((len(diagram), 1), dim)
                points = np.hstack([diagram, dim_col]).reshape(1, -1, 3)
                
                image = pi.fit_transform(points)[0]
                features[f'H{dim}_image'] = image
        else:
            # Manual implementation
            for dim, diagram in diagrams.items():
                if len(diagram) == 0:
                    features[f'H{dim}_image'] = np.zeros((n_bins, n_bins))
                    continue
                
                image = self._compute_persistence_image_manual(
                    diagram, sigma, n_bins
                )
                features[f'H{dim}_image'] = image
        
        return features
    
    def _compute_persistence_image_manual(self, diagram: np.ndarray,
                                        sigma: float, n_bins: int) -> np.ndarray:
        """
        Manually compute persistence image.
        """
        if len(diagram) == 0:
            return np.zeros((n_bins, n_bins))
        
        # Transform to birth-persistence coordinates
        births = diagram[:, 0]
        persistences = diagram[:, 1] - diagram[:, 0]
        
        # Weight by persistence
        weights = persistences
        
        # Create grid
        x_min, x_max = births.min(), births.max()
        y_min, y_max = 0, persistences.max()
        
        x_grid = np.linspace(x_min, x_max, n_bins)
        y_grid = np.linspace(y_min, y_max, n_bins)
        
        # Create image
        image = np.zeros((n_bins, n_bins))
        
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                # Sum weighted Gaussian contributions
                for k, (b, p, w) in enumerate(zip(births, persistences, weights)):
                    dist_sq = ((x - b)**2 + (y - p)**2) / (2 * sigma**2)
                    image[j, i] += w * np.exp(-dist_sq)
        
        # Normalize
        if image.max() > 0:
            image = image / image.max()
        
        return image
    
    def extract_topological_triangles(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Extract topological triangle features.
        
        These features capture geometric properties of the persistence diagram.
        """
        features = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) < 3:
                features.update({
                    f'H{dim}_triangle_area_mean': 0.0,
                    f'H{dim}_triangle_area_std': 0.0,
                    f'H{dim}_triangle_area_max': 0.0
                })
                continue
            
            # Compute triangles formed by triplets of points
            n_points = min(len(diagram), 10)  # Limit computation
            areas = []
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    for k in range(j + 1, n_points):
                        # Triangle vertices
                        p1 = diagram[i]
                        p2 = diagram[j]
                        p3 = diagram[k]
                        
                        # Compute area using cross product
                        area = 0.5 * abs(
                            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
                            (p3[0] - p1[0]) * (p2[1] - p1[1])
                        )
                        areas.append(area)
            
            if areas:
                features[f'H{dim}_triangle_area_mean'] = float(np.mean(areas))
                features[f'H{dim}_triangle_area_std'] = float(np.std(areas))
                features[f'H{dim}_triangle_area_max'] = float(np.max(areas))
            else:
                features[f'H{dim}_triangle_area_mean'] = 0.0
                features[f'H{dim}_triangle_area_std'] = 0.0
                features[f'H{dim}_triangle_area_max'] = 0.0
        
        return features
    
    def extract_persistence_vectors(self, diagrams: Dict[int, np.ndarray],
                                  n_components: int = 50) -> Dict[str, np.ndarray]:
        """
        Extract fixed-size vector representations of persistence diagrams.
        
        Uses the approach of sampling points on a grid and computing distances.
        """
        features = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                features[f'H{dim}_vector'] = np.zeros(n_components)
                continue
            
            # Create reference points on a grid
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            
            b_min, b_max = births.min(), births.max()
            d_min, d_max = deaths.min(), deaths.max()
            
            # Create grid of reference points
            n_grid = int(np.sqrt(n_components))
            b_grid = np.linspace(b_min, b_max, n_grid)
            d_grid = np.linspace(d_min, d_max, n_grid)
            
            reference_points = []
            for b in b_grid:
                for d in d_grid:
                    if d > b:  # Above diagonal only
                        reference_points.append([b, d])
            
            reference_points = np.array(reference_points)
            
            # Compute minimum distances from diagram to reference points
            if len(reference_points) > 0:
                distances = cdist(diagram, reference_points, metric='euclidean')
                min_distances = np.min(distances, axis=0)
                
                # Pad or truncate to fixed size
                if len(min_distances) < n_components:
                    vector = np.pad(min_distances, 
                                   (0, n_components - len(min_distances)))
                else:
                    vector = min_distances[:n_components]
                
                features[f'H{dim}_vector'] = vector
            else:
                features[f'H{dim}_vector'] = np.zeros(n_components)
        
        return features
    
    def extract_multi_scale_features(self, diagrams_list: List[Dict[int, np.ndarray]],
                                   scales: List[float]) -> Dict[str, float]:
        """
        Extract features across multiple scales.
        
        Parameters:
        -----------
        diagrams_list : list
            List of persistence diagrams at different scales
        scales : list
            Corresponding scale values
            
        Returns:
        --------
        dict : Multi-scale features
        """
        features = {}
        
        # Extract features at each scale
        scale_features = []
        for diagrams in diagrams_list:
            scale_feat = self.extract_statistical_features(diagrams)
            scale_features.append(scale_feat)
        
        # Compute cross-scale statistics
        feature_names = scale_features[0].keys()
        
        for feat_name in feature_names:
            values = [sf.get(feat_name, 0) for sf in scale_features]
            
            # Cross-scale statistics
            features[f'{feat_name}_scale_mean'] = float(np.mean(values))
            features[f'{feat_name}_scale_std'] = float(np.std(values))
            features[f'{feat_name}_scale_trend'] = float(
                np.polyfit(scales, values, 1)[0]  # Linear trend
            )
        
        return features