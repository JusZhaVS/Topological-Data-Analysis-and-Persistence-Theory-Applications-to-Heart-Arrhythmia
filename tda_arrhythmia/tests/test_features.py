"""
Tests for feature extraction module.
"""

import unittest
import numpy as np
from tda_arrhythmia.core.features import PersistenceFeatureExtractor


class TestPersistenceFeatureExtractor(unittest.TestCase):
    """Test cases for PersistenceFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test persistence diagrams
        self.diagrams = {
            0: np.array([[0, 0.5], [0, 1.0], [0.2, 0.8], [0.1, 0.6]]),
            1: np.array([[0.3, 0.7], [0.4, 0.9]]),
            2: np.array([[0.5, 0.6]])
        }
        
        self.empty_diagrams = {0: np.array([]), 1: np.array([])}
        
        self.extractor = PersistenceFeatureExtractor()
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = PersistenceFeatureExtractor(feature_types=['statistics', 'entropy'])
        self.assertEqual(extractor.feature_types, ['statistics', 'entropy'])
    
    def test_invalid_feature_type(self):
        """Test invalid feature type handling."""
        with self.assertRaises(ValueError):
            PersistenceFeatureExtractor(feature_types=['invalid_feature'])
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract_statistical_features(self.diagrams)
        
        # Check key features exist
        self.assertIn('latest_death_time', features)
        self.assertIn('H0_count', features)
        self.assertIn('H1_count', features)
        self.assertIn('H0_max_persistence', features)
        
        # Check values
        self.assertEqual(features['H0_count'], 4)
        self.assertEqual(features['H1_count'], 2)
        self.assertEqual(features['latest_death_time'], 1.0)  # Max death in H0
        
        # Check persistence calculations
        expected_max_pers_h0 = max(self.diagrams[0][:, 1] - self.diagrams[0][:, 0])
        self.assertAlmostEqual(features['H0_max_persistence'], expected_max_pers_h0)
        
        # Check special H0 features
        self.assertIn('H0_second_latest_death', features)
        self.assertIn('H0_death_gap', features)
    
    def test_entropy_features(self):
        """Test entropy feature extraction."""
        features = self.extractor.extract_entropy_features(self.diagrams)
        
        self.assertIn('H0_entropy', features)
        self.assertIn('H1_entropy', features)
        self.assertIn('H0_normalized_entropy', features)
        
        # Entropy should be non-negative
        self.assertGreaterEqual(features['H0_entropy'], 0)
        self.assertGreaterEqual(features['H1_entropy'], 0)
        
        # Normalized entropy should be between 0 and 1
        self.assertGreaterEqual(features['H0_normalized_entropy'], 0)
        self.assertLessEqual(features['H0_normalized_entropy'], 1)
    
    def test_topological_triangles(self):
        """Test topological triangle features."""
        features = self.extractor.extract_topological_triangles(self.diagrams)
        
        self.assertIn('H0_triangle_area_mean', features)
        self.assertIn('H0_triangle_area_std', features)
        self.assertIn('H0_triangle_area_max', features)
        
        # Areas should be non-negative
        self.assertGreaterEqual(features['H0_triangle_area_mean'], 0)
        self.assertGreaterEqual(features['H0_triangle_area_std'], 0)
        self.assertGreaterEqual(features['H0_triangle_area_max'], 0)
    
    def test_persistence_vectors(self):
        """Test persistence vector features."""
        features = self.extractor.extract_persistence_vectors(self.diagrams)
        
        self.assertIn('H0_vector', features)
        self.assertIn('H1_vector', features)
        
        # Vectors should have correct dimensions
        self.assertEqual(len(features['H0_vector']), 50)  # Default n_components
        self.assertEqual(len(features['H1_vector']), 50)
        
        # Vectors should be non-negative (distances)
        self.assertTrue(np.all(features['H0_vector'] >= 0))
        self.assertTrue(np.all(features['H1_vector'] >= 0))
    
    def test_complete_feature_extraction(self):
        """Test complete feature extraction."""
        extractor = PersistenceFeatureExtractor(
            feature_types=['statistics', 'entropy', 'triangles', 'vectors']
        )
        
        features = extractor.extract_features(self.diagrams)
        
        # Should contain features from all requested types
        self.assertIn('latest_death_time', features)  # statistics
        self.assertIn('H0_entropy', features)  # entropy
        self.assertIn('H0_triangle_area_mean', features)  # triangles
        self.assertIn('H0_vector', features)  # vectors
        
        # Check that all features are numeric
        for key, value in features.items():
            if not key.endswith('_vector'):
                self.assertIsInstance(value, (int, float, np.integer, np.floating))
    
    def test_empty_diagrams(self):
        """Test handling of empty diagrams."""
        features = self.extractor.extract_statistical_features(self.empty_diagrams)
        
        # Should handle empty diagrams gracefully
        self.assertEqual(features['H0_count'], 0)
        self.assertEqual(features['latest_death_time'], 0)
        self.assertEqual(features['H0_max_persistence'], 0)
    
    def test_multi_scale_features(self):
        """Test multi-scale feature extraction."""
        # Create multiple diagrams at different scales
        diagrams_list = [self.diagrams for _ in range(3)]
        scales = [0.5, 1.0, 2.0]
        
        features = self.extractor.extract_multi_scale_features(diagrams_list, scales)
        
        # Should contain scale-specific features
        self.assertIn('latest_death_time_scale_mean', features)
        self.assertIn('latest_death_time_scale_std', features)
        self.assertIn('latest_death_time_scale_trend', features)
        
        # Values should be reasonable
        self.assertGreaterEqual(features['latest_death_time_scale_std'], 0)
    
    def test_landscape_features_manual(self):
        """Test manual persistence landscape computation."""
        try:
            features = self.extractor.extract_landscape_features(self.diagrams)
            
            # Should contain landscape features
            for dim in [0, 1]:
                key = f'H{dim}_landscape'
                if key in features:
                    landscape = features[key]
                    self.assertIsInstance(landscape, np.ndarray)
                    self.assertGreater(len(landscape), 0)
                    
        except ImportError:
            # Skip if dependencies not available
            self.skipTest("Landscape computation dependencies not available")
    
    def test_persistence_images_manual(self):
        """Test manual persistence image computation."""
        try:
            features = self.extractor.extract_persistence_images(self.diagrams)
            
            # Should contain image features
            for dim in [0, 1]:
                key = f'H{dim}_image'
                if key in features:
                    image = features[key]
                    self.assertIsInstance(image, np.ndarray)
                    self.assertEqual(len(image.shape), 2)  # 2D image
                    self.assertTrue(np.all(image >= 0))  # Non-negative values
                    
        except ImportError:
            # Skip if dependencies not available
            self.skipTest("Image computation dependencies not available")


if __name__ == '__main__':
    unittest.main()