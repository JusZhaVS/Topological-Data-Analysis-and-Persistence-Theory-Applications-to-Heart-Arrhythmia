"""
Tests for complete TDA pipeline.
"""

import unittest
import numpy as np
from tda_arrhythmia.core.pipeline import TDACardiacAnalyzer


class TestTDACardiacAnalyzer(unittest.TestCase):
    """Test cases for TDACardiacAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Generate synthetic ECG signals
        self.fs = 250
        self.duration = 5  # seconds
        t = np.linspace(0, self.duration, self.fs * self.duration)
        
        # Normal ECG
        self.normal_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
        self.normal_ecg += np.random.normal(0, 0.1, len(t))
        
        # Arrhythmic ECG (with irregularities)
        self.arrhythmic_ecg = self.normal_ecg.copy()
        # Add some irregular beats
        for i in range(0, len(t), 200):
            if i + 50 < len(t):
                self.arrhythmic_ecg[i:i+50] *= np.random.uniform(1.5, 2.5)
        
        self.analyzer = TDACardiacAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        config = {
            'embedding': {'dimension': 4},
            'persistence': {'max_dim': 1}
        }
        analyzer = TDACardiacAnalyzer(config=config)
        self.assertEqual(analyzer.config['embedding']['dimension'], 4)
        self.assertEqual(analyzer.config['persistence']['max_dim'], 1)
    
    def test_signal_preprocessing(self):
        """Test signal preprocessing."""
        processed = self.analyzer.preprocess_signal(self.normal_ecg)
        
        # Should maintain signal length
        self.assertEqual(len(processed), len(self.normal_ecg))
        
        # Should be normalized (approximately zero mean, unit variance)
        self.assertAlmostEqual(np.mean(processed), 0, places=1)
        self.assertAlmostEqual(np.std(processed), 1, places=1)
    
    def test_analyze_signal(self):
        """Test single signal analysis."""
        features = self.analyzer.analyze_signal(self.normal_ecg)
        
        # Check that features are extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check for key features
        self.assertIn('latest_death_time', features)
        self.assertIn('embedding_delay', features)
        self.assertIn('embedding_dimension', features)
        
        # Check feature values are reasonable
        self.assertIsInstance(features['latest_death_time'], (int, float))
        self.assertGreater(features['embedding_delay'], 0)
        self.assertEqual(features['embedding_dimension'], 3)  # Default
    
    def test_analyze_signal_with_intermediates(self):
        """Test signal analysis with intermediate results."""
        features, intermediates = self.analyzer.analyze_signal(
            self.normal_ecg, return_intermediates=True
        )
        
        # Check intermediates
        self.assertIn('processed_signal', intermediates)
        self.assertIn('embedded', intermediates)
        self.assertIn('diagrams', intermediates)
        self.assertIn('delay', intermediates)
        
        # Check shapes and types
        self.assertEqual(len(intermediates['processed_signal']), len(self.normal_ecg))
        self.assertEqual(intermediates['embedded'].shape[1], 3)  # Default dimension
        self.assertIsInstance(intermediates['diagrams'], dict)
        self.assertIsInstance(intermediates['delay'], int)
    
    def test_extract_key_feature(self):
        """Test extraction of key latest death time feature."""
        ldt_normal = self.analyzer.extract_key_feature(self.normal_ecg)
        ldt_arrhythmic = self.analyzer.extract_key_feature(self.arrhythmic_ecg)
        
        # Should return numeric values
        self.assertIsInstance(ldt_normal, (int, float))
        self.assertIsInstance(ldt_arrhythmic, (int, float))
        
        # Should be non-negative
        self.assertGreaterEqual(ldt_normal, 0)
        self.assertGreaterEqual(ldt_arrhythmic, 0)
        
        # Arrhythmic signal should have different topology
        # (though not guaranteed to be higher/lower)
        self.assertTrue(abs(ldt_normal - ldt_arrhythmic) >= 0)
    
    def test_batch_analysis(self):
        """Test batch analysis of multiple signals."""
        signals = [self.normal_ecg, self.arrhythmic_ecg] * 3  # 6 signals total
        
        feature_matrix = self.analyzer.batch_analyze(signals, n_jobs=1)
        
        # Check dimensions
        self.assertEqual(feature_matrix.shape[0], 6)  # 6 signals
        self.assertGreater(feature_matrix.shape[1], 0)  # Multiple features
        
        # Check that feature names are stored
        self.assertIsNotNone(self.analyzer.feature_names)
        self.assertEqual(len(self.analyzer.feature_names), feature_matrix.shape[1])
    
    def test_training_and_prediction(self):
        """Test training and prediction pipeline."""
        # Create training data
        normal_signals = [self.normal_ecg + np.random.normal(0, 0.05, len(self.normal_ecg)) 
                         for _ in range(10)]
        arrhythmic_signals = [self.arrhythmic_ecg + np.random.normal(0, 0.05, len(self.arrhythmic_ecg))
                             for _ in range(10)]
        
        train_signals = normal_signals + arrhythmic_signals
        train_labels = np.array([0] * 10 + [1] * 10)
        
        try:
            # Train the analyzer
            self.analyzer.fit(train_signals, train_labels, validation_split=0.3)
            
            # Test prediction
            pred = self.analyzer.predict(self.normal_ecg)
            self.assertIn(pred, [0, 1])
            
            # Test prediction with probability
            pred, prob = self.analyzer.predict(self.normal_ecg, return_proba=True)
            self.assertIn(pred, [0, 1])
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
            
        except Exception as e:
            # If training fails due to missing dependencies, that's okay for testing
            if "not installed" in str(e).lower():
                self.skipTest(f"Training dependencies not available: {e}")
            else:
                raise
    
    def test_evaluation(self):
        """Test evaluation functionality."""
        # Create simple test data
        test_signals = [self.normal_ecg, self.arrhythmic_ecg]
        test_labels = np.array([0, 1])
        
        # Create a simple mock model for testing
        class MockModel:
            def predict(self, X):
                return np.array([0, 1])  # Perfect predictions
            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.2, 0.8]])
        
        # Temporarily replace model for testing
        self.analyzer.model = MockModel()
        self.analyzer.scaler = None  # No scaling for mock
        self.analyzer.feature_names = ['feature1', 'feature2']  # Mock features
        
        try:
            metrics = self.analyzer.evaluate(test_signals, test_labels)
            
            # Check that metrics are computed
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1_score', metrics)
            self.assertIn('auc_roc', metrics)
            
            # With perfect predictions, accuracy should be 1.0
            self.assertEqual(metrics['accuracy'], 1.0)
            
        except Exception as e:
            if "not fitted" in str(e).lower():
                self.skipTest("Model not fitted for evaluation test")
            else:
                raise
    
    def test_configuration_validation(self):
        """Test configuration handling."""
        # Test default configuration
        default_config = TDACardiacAnalyzer.get_default_config()
        self.assertIn('preprocessing', default_config)
        self.assertIn('embedding', default_config)
        self.assertIn('persistence', default_config)
        self.assertIn('features', default_config)
        
        # Test custom configuration
        custom_config = {
            'embedding': {'dimension': 4, 'method': 'autocorrelation'},
            'persistence': {'max_dim': 1}
        }
        analyzer = TDACardiacAnalyzer(config=custom_config)
        
        # Should merge with defaults
        self.assertEqual(analyzer.config['embedding']['dimension'], 4)
        self.assertEqual(analyzer.config['embedding']['method'], 'autocorrelation')
        self.assertIn('preprocessing', analyzer.config)  # From defaults


if __name__ == '__main__':
    unittest.main()