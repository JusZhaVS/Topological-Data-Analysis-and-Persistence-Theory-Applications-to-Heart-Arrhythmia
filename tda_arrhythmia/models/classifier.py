"""
Advanced Machine Learning Models for TDA-based Cardiac Classification
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings


class TDAClassifier(BaseEstimator, ClassifierMixin):
    """
    Specialized classifier for TDA features with topological kernel support.
    
    Implements various classifiers optimized for topological features,
    including custom kernels for persistence diagrams.
    """
    
    def __init__(self, classifier_type: str = 'ensemble',
                 use_topological_kernel: bool = False,
                 scale_features: bool = True):
        """
        Initialize TDA classifier.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier ('svm', 'xgboost', 'neural_net', 'ensemble')
        use_topological_kernel : bool
            Whether to use topological kernel for SVM
        scale_features : bool
            Whether to scale features
        """
        self.classifier_type = classifier_type
        self.use_topological_kernel = use_topological_kernel
        self.scale_features = scale_features
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           feature_names: Optional[List[str]] = None,
           **fit_params) -> 'TDAClassifier':
        """
        Fit the classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_names : list
            Names of features
        **fit_params : dict
            Additional parameters
            
        Returns:
        --------
        self : Fitted classifier
        """
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features if requested
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Fit appropriate model
        if self.classifier_type == 'svm':
            self.model = self._fit_svm(X_scaled, y)
        elif self.classifier_type == 'xgboost':
            self.model = self._fit_xgboost(X_scaled, y)
        elif self.classifier_type == 'neural_net':
            self.model = self._fit_neural_net(X_scaled, y)
        elif self.classifier_type == 'ensemble':
            self.model = self._fit_ensemble(X_scaled, y)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        self.is_fitted = True
        return self
    
    def _fit_svm(self, X: np.ndarray, y: np.ndarray):
        """Fit SVM classifier."""
        from sklearn.svm import SVC
        
        if self.use_topological_kernel:
            # Use precomputed kernel for topological features
            K = self._compute_topological_kernel_matrix(X)
            svm = SVC(kernel='precomputed', probability=True, random_state=42)
            svm.fit(K, y)
            svm._X_train = X  # Store for prediction
            return svm
        else:
            # Standard SVM with hyperparameter tuning
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
            
            svm = SVC(probability=True, random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            
            return grid_search.best_estimator_
    
    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Fit XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=42, 
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _fit_neural_net(self, X: np.ndarray, y: np.ndarray):
        """Fit neural network classifier."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        # Build model architecture optimized for TDA features
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', 
                                input_shape=(X.shape[1],),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=20, restore_best_weights=True, monitor='val_loss'
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            patience=10, factor=0.5, min_lr=1e-6
        )
        
        # Train with validation split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        return model
    
    def _fit_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble classifier."""
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.svm import SVC
        
        models = []
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        models.append(('rf', rf))
        
        # SVM
        svm = SVC(
            kernel='rbf', 
            probability=True, 
            random_state=42,
            C=10, 
            gamma='scale'
        )
        models.append(('svm', svm))
        
        # XGBoost if available
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, 
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            models.append(('xgb', xgb_model))
        except ImportError:
            warnings.warn("XGBoost not available for ensemble")
        
        # Create voting classifier
        ensemble = VotingClassifier(models, voting='soft')
        ensemble.fit(X, y)
        
        return ensemble
    
    def _compute_topological_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix for topological features.
        
        Uses a combination of RBF kernel with feature-specific weights
        based on topological importance.
        """
        from sklearn.metrics.pairwise import rbf_kernel
        
        # Feature importance weights (topological features get higher weights)
        weights = np.ones(X.shape[1])
        
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                if 'latest_death_time' in name:
                    weights[i] = 3.0  # High importance
                elif 'persistence' in name or 'entropy' in name:
                    weights[i] = 2.0  # Medium importance
                elif 'count' in name or 'birth' in name or 'death' in name:
                    weights[i] = 1.5  # Slightly higher importance
        
        # Weighted feature space
        X_weighted = X * np.sqrt(weights)
        
        # Compute RBF kernel
        gamma = 1.0 / X.shape[1]  # Default gamma
        K = rbf_kernel(X_weighted, gamma=gamma)
        
        return K
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray : Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Make predictions
        if self.classifier_type == 'svm' and self.use_topological_kernel:
            # Compute kernel with training data
            K = self._compute_topological_kernel_test(X_scaled)
            return self.model.predict(K)
        elif self.classifier_type == 'neural_net':
            proba = self.model.predict(X_scaled)
            return (proba > 0.5).astype(int).flatten()
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray : Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Make predictions
        if self.classifier_type == 'svm' and self.use_topological_kernel:
            K = self._compute_topological_kernel_test(X_scaled)
            return self.model.predict_proba(K)
        elif self.classifier_type == 'neural_net':
            proba = self.model.predict(X_scaled)
            return np.column_stack([1 - proba.flatten(), proba.flatten()])
        else:
            return self.model.predict_proba(X_scaled)
    
    def _compute_topological_kernel_test(self, X_test: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between test and training data."""
        from sklearn.metrics.pairwise import rbf_kernel
        
        X_train = self.model._X_train
        
        # Apply same weighting as in training
        weights = np.ones(X_test.shape[1])
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                if 'latest_death_time' in name:
                    weights[i] = 3.0
                elif 'persistence' in name or 'entropy' in name:
                    weights[i] = 2.0
                elif 'count' in name or 'birth' in name or 'death' in name:
                    weights[i] = 1.5
        
        X_test_weighted = X_test * np.sqrt(weights)
        X_train_weighted = X_train * np.sqrt(weights)
        
        gamma = 1.0 / X_test.shape[1]
        K = rbf_kernel(X_test_weighted, X_train_weighted, gamma=gamma)
        
        return K
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns:
        --------
        float : Accuracy score
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
        --------
        np.ndarray or None : Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        if self.classifier_type == 'xgboost':
            return self.model.feature_importances_
        elif self.classifier_type == 'ensemble':
            # Average importance from ensemble components
            importances = []
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                return np.mean(importances, axis=0)
        
        return None
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      cv: int = 5, scoring: str = 'f1') -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        dict : Cross-validation results
        """
        # Create a copy for cross-validation
        cv_classifier = TDAClassifier(
            classifier_type=self.classifier_type,
            use_topological_kernel=self.use_topological_kernel,
            scale_features=self.scale_features
        )
        
        # Perform cross-validation
        scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }


class DeepSetsClassifier(BaseEstimator, ClassifierMixin):
    """
    DeepSets architecture for persistence diagrams.
    
    This classifier can work directly with persistence diagrams
    using a permutation-invariant architecture.
    """
    
    def __init__(self, hidden_dim: int = 64, max_points: int = 100):
        """
        Initialize DeepSets classifier.
        
        Parameters:
        -----------
        hidden_dim : int
            Hidden dimension size
        max_points : int
            Maximum number of points in persistence diagram
        """
        self.hidden_dim = hidden_dim
        self.max_points = max_points
        self.model = None
        self.is_fitted = False
    
    def fit(self, persistence_diagrams: List[Dict[int, np.ndarray]], 
           y: np.ndarray) -> 'DeepSetsClassifier':
        """
        Fit the DeepSets classifier.
        
        Parameters:
        -----------
        persistence_diagrams : list
            List of persistence diagrams
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        self : Fitted classifier
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for DeepSets")
        
        # Convert diagrams to fixed-size tensors
        X = self._diagrams_to_tensor(persistence_diagrams)
        
        # Build DeepSets model
        self.model = self._build_deepsets_model()
        
        # Compile
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=20, restore_best_weights=True
        )
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_fitted = True
        return self
    
    def _diagrams_to_tensor(self, diagrams: List[Dict[int, np.ndarray]]) -> np.ndarray:
        """Convert persistence diagrams to fixed-size tensor."""
        n_samples = len(diagrams)
        
        # For simplicity, use only 0-dimensional persistence
        X = np.zeros((n_samples, self.max_points, 2))
        
        for i, diagram_dict in enumerate(diagrams):
            if 0 in diagram_dict and len(diagram_dict[0]) > 0:
                diagram = diagram_dict[0]
                n_points = min(len(diagram), self.max_points)
                
                # Sort by persistence (death - birth)
                persistences = diagram[:, 1] - diagram[:, 0]
                sorted_indices = np.argsort(persistences)[::-1]
                
                X[i, :n_points, :] = diagram[sorted_indices[:n_points]]
        
        return X
    
    def _build_deepsets_model(self):
        """Build DeepSets architecture."""
        import tensorflow as tf
        
        # Input layer
        inputs = tf.keras.Input(shape=(self.max_points, 2))
        
        # Phi network (per-point transformation)
        phi = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
        phi = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(phi)
        phi = tf.keras.layers.Dense(self.hidden_dim)(phi)
        
        # Aggregation (sum over points)
        aggregated = tf.keras.layers.GlobalAveragePooling1D()(phi)
        
        # Rho network (after aggregation)
        rho = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(aggregated)
        rho = tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu')(rho)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(rho)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def predict(self, persistence_diagrams: List[Dict[int, np.ndarray]]) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted")
        
        X = self._diagrams_to_tensor(persistence_diagrams)
        proba = self.model.predict(X)
        return (proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, persistence_diagrams: List[Dict[int, np.ndarray]]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted")
        
        X = self._diagrams_to_tensor(persistence_diagrams)
        proba = self.model.predict(X)
        return np.column_stack([1 - proba.flatten(), proba.flatten()])