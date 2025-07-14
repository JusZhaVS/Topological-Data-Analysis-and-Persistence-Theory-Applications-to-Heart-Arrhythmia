"""
Complete TDA Pipeline for Cardiac Arrhythmia Analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal as sp_signal
import warnings

from ..core.embedding import TakensEmbedding
from ..core.persistence import PersistenceComputer
from ..core.features import PersistenceFeatureExtractor


class TDACardiacAnalyzer:
    """
    Complete TDA pipeline for cardiac arrhythmia detection.
    
    Implements the full pipeline described in the technical guide:
    1. Signal preprocessing
    2. Takens' embedding with mutual information
    3. Vietoris-Rips persistence computation
    4. Feature extraction (including latest death time)
    5. Classification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TDA cardiac analyzer.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config or self.get_default_config()
        
        # Initialize components
        self.embedder = TakensEmbedding(
            dimension=self.config['embedding']['dimension'],
            method=self.config['embedding']['method'],
            max_lag=self.config['embedding']['max_delay']
        )
        
        self.persistence_computer = PersistenceComputer(
            library=self.config['persistence']['library'],
            max_dimension=self.config['persistence']['max_dim'],
            max_edge_length=self.config['persistence']['max_edge_length']
        )
        
        self.feature_extractor = PersistenceFeatureExtractor(
            feature_types=self.config['features']['types']
        )
        
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    @staticmethod
    def get_default_config():
        """Get default configuration."""
        return {
            'preprocessing': {
                'denoise': True,
                'normalize': True,
                'fs': 250,  # Sampling frequency
                'baseline_filter': True,
                'powerline_filter': True
            },
            'embedding': {
                'method': 'mutual_information',
                'dimension': 3,
                'max_delay': 50
            },
            'persistence': {
                'library': 'ripser',
                'max_dim': 2,
                'max_edge_length': 2.0
            },
            'features': {
                'types': ['statistics', 'entropy', 'landscapes'],
                'use_multiscale': True,
                'scales': [0.5, 1.0, 2.0, 4.0]
            },
            'classifier': {
                'type': 'ensemble',
                'models': ['svm', 'xgboost', 'random_forest']
            }
        }
    
    def preprocess_signal(self, signal: np.ndarray, fs: Optional[float] = None) -> np.ndarray:
        """
        Preprocess ECG signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Raw ECG signal
        fs : float
            Sampling frequency (Hz)
            
        Returns:
        --------
        np.ndarray : Preprocessed signal
        """
        fs = fs or self.config['preprocessing']['fs']
        processed = signal.copy()
        
        # Remove baseline wander
        if self.config['preprocessing']['baseline_filter']:
            # High-pass filter at 0.5 Hz
            b, a = sp_signal.butter(4, 0.5/(fs/2), btype='highpass')
            processed = sp_signal.filtfilt(b, a, processed)
        
        # Remove powerline interference
        if self.config['preprocessing']['powerline_filter']:
            # Notch filters at 50 and 60 Hz
            for freq in [50, 60]:
                if freq < fs/2:  # Nyquist check
                    b, a = sp_signal.iirnotch(freq, 30, fs)
                    processed = sp_signal.filtfilt(b, a, processed)
        
        # Denoising
        if self.config['preprocessing']['denoise']:
            processed = self._wavelet_denoise(processed)
        
        # Normalization
        if self.config['preprocessing']['normalize']:
            mean = np.mean(processed)
            std = np.std(processed)
            if std > 0:
                processed = (processed - mean) / std
        
        return processed
    
    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising."""
        try:
            import pywt
        except ImportError:
            warnings.warn("PyWavelets not installed. Skipping wavelet denoising.")
            return signal
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        
        # Estimate noise level (using detail coefficients at finest level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Soft thresholding
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [
            pywt.threshold(c, threshold, mode='soft') 
            for c in coeffs_thresh[1:]
        ]
        
        # Reconstruction
        denoised = pywt.waverec(coeffs_thresh, 'db4')
        
        # Handle length mismatch
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        
        return denoised
    
    def analyze_signal(self, signal: np.ndarray, 
                      return_intermediates: bool = False) -> Union[Dict, Tuple]:
        """
        Perform complete TDA analysis on ECG signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            ECG signal
        return_intermediates : bool
            Whether to return intermediate results
            
        Returns:
        --------
        dict : Extracted features (or tuple with intermediates)
        """
        # Preprocess
        processed = self.preprocess_signal(signal)
        
        # Takens embedding
        embedded, delay = self.embedder.embed(processed)
        
        # Compute persistence
        diagrams = self.persistence_computer.compute_persistence(embedded)
        
        # Extract features
        features = self.feature_extractor.extract_features(diagrams)
        
        # Add embedding parameters as features
        features['embedding_delay'] = delay
        features['embedding_dimension'] = self.embedder.dimension
        
        # Multi-scale analysis if enabled
        if self.config['features']['use_multiscale']:
            multiscale_features = self._multiscale_analysis(processed)
            features.update(multiscale_features)
        
        if return_intermediates:
            return features, {
                'processed_signal': processed,
                'embedded': embedded,
                'diagrams': diagrams,
                'delay': delay
            }
        
        return features
    
    def _multiscale_analysis(self, signal: np.ndarray) -> Dict[str, float]:
        """Perform multi-scale TDA analysis."""
        scales = self.config['features']['scales']
        multiscale_features = {}
        
        for scale in scales:
            # Apply scale-specific processing
            if scale < 1.0:
                # Upsample for fine-scale analysis
                from scipy import signal as sp_signal
                n_samples = int(len(signal) / scale)
                scaled_signal = sp_signal.resample(signal, n_samples)
            else:
                # Smooth for coarse-scale analysis
                from scipy.ndimage import gaussian_filter1d
                scaled_signal = gaussian_filter1d(signal, sigma=scale)
            
            # Compute TDA features at this scale
            try:
                embedded, _ = self.embedder.embed(scaled_signal)
                diagrams = self.persistence_computer.compute_persistence(embedded)
                scale_features = self.feature_extractor.extract_statistical_features(diagrams)
                
                # Add scale prefix to feature names
                for feat_name, value in scale_features.items():
                    multiscale_features[f'{feat_name}_scale_{scale}'] = value
                    
            except Exception as e:
                warnings.warn(f"Failed to compute features at scale {scale}: {e}")
                continue
        
        return multiscale_features
    
    def extract_key_feature(self, signal: np.ndarray) -> float:
        """
        Extract the key "latest death time" feature.
        
        This is the primary feature mentioned in the technical guide
        that achieves high accuracy for arrhythmia detection.
        
        Parameters:
        -----------
        signal : np.ndarray
            ECG signal
            
        Returns:
        --------
        float : Latest death time of 0-dimensional features
        """
        # Preprocess
        processed = self.preprocess_signal(signal)
        
        # Embed
        embedded, _ = self.embedder.embed(processed)
        
        # Compute persistence
        diagrams = self.persistence_computer.compute_persistence(embedded)
        
        # Extract latest death time
        if 0 in diagrams and len(diagrams[0]) > 0:
            deaths = diagrams[0][:, 1]
            return float(np.max(deaths))
        else:
            return 0.0
    
    def batch_analyze(self, signals: List[np.ndarray], 
                     n_jobs: int = -1) -> np.ndarray:
        """
        Analyze multiple signals in parallel.
        
        Parameters:
        -----------
        signals : list
            List of ECG signals
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
            
        Returns:
        --------
        np.ndarray : Feature matrix
        """
        from joblib import Parallel, delayed
        
        # Process in parallel
        features_list = Parallel(n_jobs=n_jobs)(
            delayed(self.analyze_signal)(signal) 
            for signal in signals
        )
        
        # Convert to matrix
        if not features_list:
            return np.array([])
        
        # Get feature names from first result
        self.feature_names = list(features_list[0].keys())
        
        # Create feature matrix
        n_samples = len(features_list)
        n_features = len(self.feature_names)
        feature_matrix = np.zeros((n_samples, n_features))
        
        for i, features in enumerate(features_list):
            for j, feat_name in enumerate(self.feature_names):
                value = features.get(feat_name, 0)
                # Handle array features (flatten them)
                if isinstance(value, np.ndarray):
                    value = value.flatten()[0] if len(value) > 0 else 0
                feature_matrix[i, j] = float(value)
        
        return feature_matrix
    
    def fit(self, signals: List[np.ndarray], labels: np.ndarray,
            validation_split: float = 0.2) -> 'TDACardiacAnalyzer':
        """
        Train the complete pipeline.
        
        Parameters:
        -----------
        signals : list
            List of ECG signals
        labels : np.ndarray
            Binary labels (0: normal, 1: arrhythmia)
        validation_split : float
            Fraction for validation
            
        Returns:
        --------
        self : Trained analyzer
        """
        # Extract features
        print("Extracting features...")
        X = self.batch_analyze(signals)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=validation_split, 
            random_state=42, stratify=labels
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train classifier
        print("Training classifier...")
        self.model = self._train_classifier(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Evaluate
        from sklearn.metrics import accuracy_score, classification_report
        val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"\nValidation Accuracy: {val_acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_pred, 
                                  target_names=['Normal', 'Arrhythmia']))
        
        return self
    
    def _train_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray):
        """Train classifier based on configuration."""
        classifier_type = self.config['classifier']['type']
        
        if classifier_type == 'ensemble':
            return self._train_ensemble(X_train, y_train, X_val, y_val)
        elif classifier_type == 'svm':
            return self._train_svm(X_train, y_train)
        elif classifier_type == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_val, y_val)
        elif classifier_type == 'neural_net':
            return self._train_neural_net(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def _train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray):
        """Train ensemble of classifiers."""
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.svm import SVC
        
        models = []
        
        if 'svm' in self.config['classifier']['models']:
            svm = SVC(kernel='rbf', probability=True, random_state=42)
            models.append(('svm', svm))
        
        if 'random_forest' in self.config['classifier']['models']:
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            models.append(('rf', rf))
        
        if 'xgboost' in self.config['classifier']['models']:
            try:
                import xgboost as xgb
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100, random_state=42, 
                    use_label_encoder=False, eval_metric='logloss'
                )
                models.append(('xgb', xgb_model))
            except ImportError:
                warnings.warn("XGBoost not installed")
        
        # Create voting classifier
        ensemble = VotingClassifier(models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def _train_svm(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train SVM classifier."""
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # Grid search for best parameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """Train XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train with early stopping
        model = xgb.train(
            params, dtrain, 
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Return sklearn-compatible wrapper
        sklearn_model = xgb.XGBClassifier(**params)
        sklearn_model._Booster = model
        
        return sklearn_model
    
    def _train_neural_net(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray):
        """Train neural network classifier."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed")
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', 
                                input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Train with early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=20, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        return model
    
    def predict(self, signal: np.ndarray, 
               return_proba: bool = False) -> Union[int, Tuple[int, float]]:
        """
        Predict arrhythmia for a single signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            ECG signal
        return_proba : bool
            Whether to return probability
            
        Returns:
        --------
        int or tuple : Prediction (and probability if requested)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features
        features = self.analyze_signal(signal)
        
        # Convert to feature vector
        feature_vector = np.zeros(len(self.feature_names))
        for i, feat_name in enumerate(self.feature_names):
            value = features.get(feat_name, 0)
            if isinstance(value, np.ndarray):
                value = value.flatten()[0] if len(value) > 0 else 0
            feature_vector[i] = float(value)
        
        # Scale
        feature_vector_scaled = self.scaler.transform(
            feature_vector.reshape(1, -1)
        )
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(feature_vector_scaled)[0, 1]
            pred = int(proba > 0.5)
        else:
            # Neural network
            proba = float(self.model.predict(feature_vector_scaled)[0, 0])
            pred = int(proba > 0.5)
        
        if return_proba:
            return pred, proba
        return pred
    
    def evaluate(self, signals: List[np.ndarray], labels: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        signals : list
            Test signals
        labels : np.ndarray
            True labels
            
        Returns:
        --------
        dict : Performance metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        # Get predictions
        predictions = []
        probabilities = []
        
        for signal in signals:
            pred, proba = self.predict(signal, return_proba=True)
            predictions.append(pred)
            probabilities.append(proba)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'auc_roc': roc_auc_score(labels, probabilities)
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['specificity'] = tn / (tn + fp)
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['confusion_matrix'] = cm
        
        return metrics