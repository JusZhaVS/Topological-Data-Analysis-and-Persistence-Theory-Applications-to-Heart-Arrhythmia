"""
Tests for persistence computation module.
"""

import unittest
import numpy as np
from tda_arrhythmia.core.persistence import PersistenceComputer


class TestPersistenceComputer(unittest.TestCase):
    """Test cases for PersistenceComputer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create a simple point cloud (circle)
        n_points = 50
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        self.point_cloud = np.column_stack([
            np.cos(theta) + 0.1 * np.random.randn(n_points),
            np.sin(theta) + 0.1 * np.random.randn(n_points)
        ])
        
        # Add some noise points
        noise_points = np.random.randn(10, 2) * 0.1
        self.point_cloud = np.vstack([self.point_cloud, noise_points])
    
    def test_initialization(self):
        """Test computer initialization."""
        computer = PersistenceComputer(library='ripser', max_dimension=2)
        self.assertEqual(computer.library, 'ripser')
        self.assertEqual(computer.max_dimension, 2)
    
    def test_persistence_computation_fallback(self):
        """Test persistence computation with fallback."""
        # This should work even without ripser installed
        computer = PersistenceComputer(library='ripser')
        
        try:
            diagrams = computer.compute_persistence(self.point_cloud)
            
            # Check that we get diagrams
            self.assertIsInstance(diagrams, dict)
            
            # Should have at least 0-dimensional features
            self.assertIn(0, diagrams)
            
            # Check format of diagrams
            for dim, diagram in diagrams.items():
                if len(diagram) > 0:
                    self.assertEqual(diagram.shape[1], 2)  # birth, death
                    # Birth should be <= death
                    self.assertTrue(np.all(diagram[:, 0] <= diagram[:, 1]))
                    
        except ImportError:
            # If ripser not available, test should still pass
            self.skipTest("Ripser not available")
    
    def test_betti_curves(self):
        """Test Betti curve computation."""
        computer = PersistenceComputer()
        
        # Create simple diagrams for testing
        diagrams = {
            0: np.array([[0, 0.5], [0, 1.0], [0.2, 0.8]]),
            1: np.array([[0.3, 0.7]])
        }
        
        betti_curves = computer.compute_betti_curves(diagrams)
        
        self.assertIsInstance(betti_curves, dict)
        
        for dim in [0, 1]:
            self.assertIn(dim, betti_curves)
            filtration_values, betti_numbers = betti_curves[dim]
            
            self.assertEqual(len(filtration_values), len(betti_numbers))
            self.assertTrue(np.all(betti_numbers >= 0))
    
    def test_persistence_statistics(self):
        """Test statistical feature computation."""
        computer = PersistenceComputer()
        
        # Create test diagrams
        diagrams = {
            0: np.array([[0, 0.5], [0, 1.0], [0.2, 0.8]]),
            1: np.array([[0.3, 0.7], [0.1, 0.6]])
        }
        
        stats = computer.compute_persistence_statistics(diagrams)
        
        # Check required features
        self.assertIn('H0_count', stats)
        self.assertIn('H1_count', stats)
        self.assertIn('latest_death_time', stats)
        
        # Check values
        self.assertEqual(stats['H0_count'], 3)
        self.assertEqual(stats['H1_count'], 2)
        self.assertEqual(stats['latest_death_time'], 1.0)  # Max death in H0
        
        # Check statistical features
        self.assertIn('H0_mean_persistence', stats)
        self.assertIn('H0_max_persistence', stats)
        self.assertGreater(stats['H0_max_persistence'], 0)
    
    def test_persistence_entropy(self):
        """Test persistence entropy computation."""
        computer = PersistenceComputer()
        
        diagrams = {
            0: np.array([[0, 0.5], [0, 1.0], [0.2, 0.8]]),
            1: np.array([[0.3, 0.7]])
        }
        
        entropies = computer.compute_persistence_entropy(diagrams)
        
        self.assertIn('H0_entropy', entropies)
        self.assertIn('H1_entropy', entropies)
        
        # Entropy should be non-negative
        self.assertGreaterEqual(entropies['H0_entropy'], 0)
        self.assertGreaterEqual(entropies['H1_entropy'], 0)
    
    def test_empty_diagrams(self):
        """Test handling of empty persistence diagrams."""
        computer = PersistenceComputer()
        
        # Empty diagrams
        diagrams = {0: np.array([]), 1: np.array([])}
        
        stats = computer.compute_persistence_statistics(diagrams)
        entropies = computer.compute_persistence_entropy(diagrams)
        
        # Should handle empty diagrams gracefully
        self.assertEqual(stats['H0_count'], 0)
        self.assertEqual(stats['latest_death_time'], 0)
        self.assertEqual(entropies['H0_entropy'], 0)


if __name__ == '__main__':
    unittest.main()