"""
Visualization Utilities for TDA Cardiac Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings


class TDAVisualizer:
    """
    Comprehensive visualization tools for TDA analysis results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize TDA visualizer.
        
        Parameters:
        -----------
        style : str
            Matplotlib style
        figsize : tuple
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color palettes
        self.colors = {
            'H0': '#FF6B6B',  # Red for 0-dimensional
            'H1': '#4ECDC4',  # Teal for 1-dimensional  
            'H2': '#45B7D1',  # Blue for 2-dimensional
            'normal': '#95E1D3',
            'arrhythmia': '#F38BA8'
        }
    
    def plot_signal_and_embedding(self, signal: np.ndarray, embedded: np.ndarray,
                                 title: str = "ECG Signal and Phase Space Embedding",
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot original signal and its phase space embedding.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original time series
        embedded : np.ndarray
            Embedded phase space
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        figsize = figsize or self.figsize
        
        if embedded.shape[1] == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig = plt.figure(figsize=(15, 8))
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2, projection='3d')
            ax3 = plt.subplot(1, 3, 3)
        
        # Original signal
        ax1.plot(signal, 'b-', linewidth=1)
        ax1.set_title('Original ECG Signal')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Phase space embedding
        if embedded.shape[1] == 2:
            ax2.plot(embedded[:, 0], embedded[:, 1], 'r-', alpha=0.7, linewidth=0.8)
            ax2.scatter(embedded[0, 0], embedded[0, 1], c='g', s=50, label='Start')
            ax2.scatter(embedded[-1, 0], embedded[-1, 1], c='r', s=50, label='End')
            ax2.set_xlabel('x(t)')
            ax2.set_ylabel('x(t+τ)')
            ax2.set_title('2D Phase Space Embedding')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # 3D embedding
            ax2.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], 
                    'r-', alpha=0.7, linewidth=0.8)
            ax2.scatter(embedded[0, 0], embedded[0, 1], embedded[0, 2], 
                       c='g', s=50, label='Start')
            ax2.scatter(embedded[-1, 0], embedded[-1, 1], embedded[-1, 2], 
                       c='r', s=50, label='End')
            ax2.set_xlabel('x(t)')
            ax2.set_ylabel('x(t+τ)')
            ax2.set_zlabel('x(t+2τ)')
            ax2.set_title('3D Phase Space Embedding')
            ax2.legend()
            
            # 2D projection for comparison
            ax3.plot(embedded[:, 0], embedded[:, 1], 'b-', alpha=0.7, linewidth=0.8)
            ax3.set_xlabel('x(t)')
            ax3.set_ylabel('x(t+τ)')
            ax3.set_title('2D Projection')
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_persistence_diagram(self, diagrams: Dict[int, np.ndarray],
                               title: str = "Persistence Diagram",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot persistence diagrams for multiple homology dimensions.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each dimension
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                continue
            
            color = self.colors.get(f'H{dim}', f'C{dim}')
            marker = 'o' if dim == 0 else 's' if dim == 1 else '^'
            
            ax.scatter(diagram[:, 0], diagram[:, 1], 
                      c=color, marker=marker, s=60, alpha=0.7,
                      label=f'H{dim} ({len(diagram)} features)')
        
        # Plot diagonal
        all_values = []
        for diagram in diagrams.values():
            if len(diagram) > 0:
                all_values.extend(diagram.flatten())
        
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
                   alpha=0.5, linewidth=2, label='Diagonal')
        
        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig
    
    def plot_barcode(self, diagrams: Dict[int, np.ndarray],
                    title: str = "Persistence Barcode",
                    figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot persistence barcode.
        
        Parameters:
        -----------
        diagrams : dict
            Persistence diagrams by dimension
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        # Plot bars for each dimension
        for dim in sorted(diagrams.keys()):
            diagram = diagrams[dim]
            if len(diagram) == 0:
                continue
            
            color = self.colors.get(f'H{dim}', f'C{dim}')
            
            # Sort by birth time
            sorted_indices = np.argsort(diagram[:, 0])
            sorted_diagram = diagram[sorted_indices]
            
            for i, (birth, death) in enumerate(sorted_diagram):
                ax.barh(y_pos, death - birth, left=birth, 
                       height=0.8, color=color, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
                y_pos += 1
            
            # Add dimension label
            if len(diagram) > 0:
                y_ticks.append(y_pos - len(diagram)/2 - 0.5)
                y_labels.append(f'H{dim}')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.set_xlabel('Filtration Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        return fig
    
    def plot_betti_curves(self, betti_curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
                         title: str = "Betti Curves",
                         figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot Betti curves (number of features vs filtration value).
        
        Parameters:
        -----------
        betti_curves : dict
            Betti curves by dimension
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        for dim, (filtration_values, betti_numbers) in betti_curves.items():
            color = self.colors.get(f'H{dim}', f'C{dim}')
            ax.plot(filtration_values, betti_numbers, 
                   color=color, linewidth=2, label=f'β{dim}')
            ax.fill_between(filtration_values, betti_numbers, 
                           alpha=0.3, color=color)
        
        ax.set_xlabel('Filtration Value', fontsize=12)
        ax.set_ylabel('Betti Number', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_feature_comparison(self, features_normal: Dict[str, float],
                               features_arrhythmia: Dict[str, float],
                               feature_names: Optional[List[str]] = None,
                               title: str = "TDA Feature Comparison",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Compare TDA features between normal and arrhythmic signals.
        
        Parameters:
        -----------
        features_normal : dict
            Features from normal signals
        features_arrhythmia : dict
            Features from arrhythmic signals
        feature_names : list
            Specific features to plot
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        if feature_names is None:
            # Select key features
            feature_names = [
                'latest_death_time', 'H0_count', 'H0_max_persistence',
                'H1_count', 'H1_max_persistence', 'H0_entropy'
            ]
            # Filter to available features
            feature_names = [f for f in feature_names 
                           if f in features_normal and f in features_arrhythmia]
        
        if not feature_names:
            raise ValueError("No common features found")
        
        figsize = figsize or (12, 8)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names[:6]):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Box plot comparison
            data = [
                [features_normal[feature]] if isinstance(features_normal[feature], (int, float)) 
                else features_normal[feature],
                [features_arrhythmia[feature]] if isinstance(features_arrhythmia[feature], (int, float))
                else features_arrhythmia[feature]
            ]
            
            bp = ax.boxplot(data, labels=['Normal', 'Arrhythmia'],
                          patch_artist=True)
            
            bp['boxes'][0].set_facecolor(self.colors['normal'])
            bp['boxes'][1].set_facecolor(self.colors['arrhythmia'])
            
            ax.set_title(feature.replace('_', ' ').title(), fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(feature_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_classification_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: Optional[np.ndarray] = None,
                                   title: str = "Classification Results",
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot classification results including confusion matrix and ROC curve.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_scores : np.ndarray
            Prediction scores/probabilities
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        if y_scores is not None:
            figsize = figsize or (15, 5)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        else:
            figsize = figsize or (10, 5)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Arrhythmia'],
                   yticklabels=['Normal', 'Arrhythmia'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        
        bars = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        ax2.set_ylim(0, 1)
        ax2.set_title('Classification Metrics')
        ax2.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # ROC curve (if scores provided)
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            ax3.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve')
            ax3.legend(loc="lower right")
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_mutual_information(self, signal: np.ndarray, max_lag: int = 50,
                               title: str = "Mutual Information Analysis",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot mutual information vs time delay for optimal embedding.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        max_lag : int
            Maximum lag to compute
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        from sklearn.feature_selection import mutual_info_regression
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute mutual information
        lags = range(1, min(max_lag, len(signal) // 4))
        mi_values = []
        
        for lag in lags:
            x = signal[:-lag].reshape(-1, 1)
            y = signal[lag:]
            mi = mutual_info_regression(x, y, random_state=42)[0]
            mi_values.append(mi)
        
        # Plot
        ax.plot(lags, mi_values, 'b-', linewidth=2, marker='o', markersize=4)
        
        # Find and mark optimal delay (first minimum)
        if len(mi_values) > 2:
            for i in range(1, len(mi_values) - 1):
                if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                    optimal_delay = lags[i]
                    ax.axvline(x=optimal_delay, color='r', linestyle='--', 
                              linewidth=2, label=f'Optimal delay = {optimal_delay}')
                    break
        
        ax.set_xlabel('Time Delay (samples)', fontsize=12)
        ax.set_ylabel('Mutual Information', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_pipeline_results(self, signal: np.ndarray, embedded: np.ndarray,
                             diagrams: Dict[int, np.ndarray], features: Dict,
                             prediction: Optional[int] = None,
                             title: str = "Complete TDA Analysis Pipeline") -> plt.Figure:
        """
        Plot complete pipeline results in a comprehensive view.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        embedded : np.ndarray
            Embedded signal
        diagrams : dict
            Persistence diagrams
        features : dict
            Extracted features
        prediction : int
            Classification prediction
        title : str
            Overall title
            
        Returns:
        --------
        matplotlib.Figure : The figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Original signal
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(signal, 'b-', linewidth=1)
        ax1.set_title('ECG Signal')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Phase space embedding
        ax2 = plt.subplot(2, 3, 2)
        if embedded.shape[1] >= 2:
            ax2.plot(embedded[:, 0], embedded[:, 1], 'r-', alpha=0.7, linewidth=0.8)
            ax2.set_xlabel('x(t)')
            ax2.set_ylabel('x(t+τ)')
            ax2.set_title('Phase Space Embedding')
            ax2.grid(True, alpha=0.3)
        
        # Persistence diagram
        ax3 = plt.subplot(2, 3, 3)
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                continue
            color = self.colors.get(f'H{dim}', f'C{dim}')
            marker = 'o' if dim == 0 else 's'
            ax3.scatter(diagram[:, 0], diagram[:, 1], 
                       c=color, marker=marker, s=40, alpha=0.7,
                       label=f'H{dim}')
        
        # Diagonal
        all_values = []
        for diagram in diagrams.values():
            if len(diagram) > 0:
                all_values.extend(diagram.flatten())
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax3.set_xlabel('Birth')
        ax3.set_ylabel('Death')
        ax3.set_title('Persistence Diagram')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Key features
        ax4 = plt.subplot(2, 3, 4)
        key_features = ['latest_death_time', 'H0_count', 'H0_max_persistence', 
                       'H1_count', 'H0_entropy']
        key_features = [f for f in key_features if f in features]
        
        if key_features:
            values = [features[f] for f in key_features]
            bars = ax4.bar(range(len(key_features)), values, 
                          color='lightblue', edgecolor='navy')
            ax4.set_xticks(range(len(key_features)))
            ax4.set_xticklabels([f.replace('_', '\n') for f in key_features], 
                               rotation=0, fontsize=8)
            ax4.set_title('Key TDA Features')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Barcode
        ax5 = plt.subplot(2, 3, 5)
        y_pos = 0
        for dim in sorted(diagrams.keys()):
            diagram = diagrams[dim]
            if len(diagram) == 0:
                continue
            color = self.colors.get(f'H{dim}', f'C{dim}')
            for birth, death in diagram:
                ax5.barh(y_pos, death - birth, left=birth, 
                        height=0.8, color=color, alpha=0.7)
                y_pos += 1
        
        ax5.set_xlabel('Filtration Value')
        ax5.set_title('Persistence Barcode')
        ax5.grid(True, axis='x', alpha=0.3)
        
        # Prediction result
        ax6 = plt.subplot(2, 3, 6)
        if prediction is not None:
            colors = [self.colors['normal'], self.colors['arrhythmia']]
            labels = ['Normal', 'Arrhythmia']
            
            # Create pie chart showing prediction
            sizes = [1-prediction, prediction] if prediction in [0, 1] else [0.5, 0.5]
            ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90)
            ax6.set_title(f'Prediction: {labels[prediction]}' if prediction in [0, 1] 
                         else 'Prediction: Unknown')
        else:
            ax6.text(0.5, 0.5, 'No Prediction', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Classification')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig