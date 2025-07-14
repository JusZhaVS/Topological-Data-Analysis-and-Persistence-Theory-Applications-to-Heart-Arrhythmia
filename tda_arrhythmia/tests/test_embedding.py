"""
Tests for Takens' embedding module.
"""

import unittest
import numpy as np
from tda_arrhythmia.core.embedding import TakensEmbedding


class TestTakensEmbedding(unittest.TestCase):
    """Test cases for TakensEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Generate test signal
        t = np.linspace(0, 10, 1000)
        self.signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
        self.signal += np.random.normal(0, 0.1, len(self.signal))
        
        self.embedder = TakensEmbedding()
    
    def test_initialization(self):
        """Test embedder initialization."""
        embedder = TakensEmbedding(dimension=3, delay=5)
        self.assertEqual(embedder.dimension, 3)
        self.assertEqual(embedder.delay, 5)
        self.assertEqual(embedder.method, 'mutual_information')
    
    def test_mutual_information_delay(self):
        """Test mutual information delay calculation."""
        delay = self.embedder.find_optimal_delay(self.signal)
        self.assertIsInstance(delay, int)
        self.assertGreater(delay, 0)
        self.assertLess(delay, len(self.signal) // 4)
    
    def test_autocorrelation_delay(self):
        """Test autocorrelation delay calculation."""
        embedder = TakensEmbedding(method='autocorrelation')
        delay = embedder.find_optimal_delay(self.signal)
        self.assertIsInstance(delay, int)
        self.assertGreater(delay, 0)
    
    def test_embedding_basic(self):
        """Test basic embedding functionality."""
        embedded, delay = self.embedder.embed(self.signal)
        
        # Check dimensions
        self.assertEqual(embedded.shape[1], self.embedder.dimension)
        expected_length = len(self.signal) - (self.embedder.dimension - 1) * delay
        self.assertEqual(embedded.shape[0], expected_length)
        
        # Check that embedding is not trivial
        self.assertGreater(np.std(embedded[:, 0]), 0)
        self.assertGreater(np.std(embedded[:, 1]), 0)
    
    def test_embedding_with_specified_delay(self):
        """Test embedding with user-specified delay."""
        delay = 5
        embedded, returned_delay = self.embedder.embed(self.signal, delay=delay)
        
        self.assertEqual(returned_delay, delay)
        expected_length = len(self.signal) - (self.embedder.dimension - 1) * delay
        self.assertEqual(embedded.shape[0], expected_length)
    
    def test_embedding_dimensions(self):
        """Test different embedding dimensions."""
        for dim in [2, 3, 4]:
            embedder = TakensEmbedding(dimension=dim)
            embedded, _ = embedder.embed(self.signal)
            self.assertEqual(embedded.shape[1], dim)
    
    def test_short_signal_error(self):
        """Test error handling for short signals."""
        short_signal = np.random.randn(10)
        embedder = TakensEmbedding(dimension=5, delay=10)
        
        with self.assertRaises(ValueError):
            embedder.embed(short_signal)
    
    def test_fit_transform_interface(self):
        """Test sklearn-compatible interface."""
        embedded = self.embedder.fit_transform(self.signal)
        self.assertEqual(embedded.shape[1], self.embedder.dimension)
        self.assertGreater(embedded.shape[0], 0)
    
    def test_optimal_parameters(self):
        """Test optimal parameter selection."""
        optimal_params = self.embedder.get_optimal_parameters(self.signal)
        
        self.assertIn('dimension', optimal_params)
        self.assertIn('delay', optimal_params)
        self.assertIn('fnn_percentages', optimal_params)
        
        self.assertIsInstance(optimal_params['dimension'], int)
        self.assertIsInstance(optimal_params['delay'], int)
        self.assertGreater(optimal_params['dimension'], 1)
        self.assertGreater(optimal_params['delay'], 0)


if __name__ == '__main__':
    unittest.main()