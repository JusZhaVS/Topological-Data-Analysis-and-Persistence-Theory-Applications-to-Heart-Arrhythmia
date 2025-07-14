#!/usr/bin/env python3
"""
Quickstart Example: TDA for Heart Arrhythmia Detection

This example demonstrates the basic usage of the TDA pipeline
for cardiac arrhythmia detection, following the methodology
described in the technical guide.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import our TDA modules
from tda_arrhythmia import TDACardiacAnalyzer, PhysioNetLoader, TDAVisualizer


def generate_synthetic_ecg(length: int = 2500, fs: int = 250, 
                          arrhythmia: bool = False) -> np.ndarray:
    """
    Generate synthetic ECG signal for demonstration.
    
    Parameters:
    -----------
    length : int
        Signal length in samples
    fs : int
        Sampling frequency
    arrhythmia : bool
        Whether to include arrhythmic patterns
        
    Returns:
    --------
    np.ndarray : Synthetic ECG signal
    """
    t = np.linspace(0, length/fs, length)
    
    # Base ECG components
    heart_rate = 75  # BPM
    rr_interval = 60 / heart_rate  # seconds
    
    signal = np.zeros_like(t)
    
    # Generate QRS complexes
    for beat_time in np.arange(0, t[-1], rr_interval):
        # Add some variability
        beat_time += np.random.normal(0, 0.02)
        
        if beat_time > t[-1]:
            break
        
        # QRS complex (simplified)
        qrs_width = 0.1  # seconds
        qrs_indices = np.abs(t - beat_time) < qrs_width/2
        
        if arrhythmia and np.random.random() < 0.3:
            # Premature ventricular contraction (PVC)
            amplitude = np.random.uniform(1.5, 2.5)
            width_factor = np.random.uniform(1.5, 2.0)
        else:
            # Normal beat
            amplitude = 1.0
            width_factor = 1.0
        
        # Gaussian-like QRS
        qrs_signal = amplitude * np.exp(-((t - beat_time) / (qrs_width * width_factor))**2 * 20)
        signal += qrs_signal
    
    # Add noise
    noise_level = 0.05
    signal += np.random.normal(0, noise_level, length)
    
    return signal


def main():
    """Main example function."""
    print("TDA Cardiac Arrhythmia Detection - Quickstart Example")
    print("=" * 60)
    
    # Initialize components
    analyzer = TDACardiacAnalyzer()
    visualizer = TDAVisualizer()
    
    # Generate synthetic data
    print("\n1. Generating synthetic ECG data...")
    
    # Normal signals
    normal_signals = [generate_synthetic_ecg(arrhythmia=False) for _ in range(50)]
    
    # Arrhythmic signals  
    arrhythmic_signals = [generate_synthetic_ecg(arrhythmia=True) for _ in range(50)]
    
    # Combine data
    all_signals = normal_signals + arrhythmic_signals
    labels = np.array([0] * 50 + [1] * 50)  # 0: normal, 1: arrhythmia
    
    print(f"Generated {len(normal_signals)} normal and {len(arrhythmic_signals)} arrhythmic signals")
    
    # Split data
    train_signals, test_signals, y_train, y_test = train_test_split(
        all_signals, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_signals)} signals")
    print(f"Test set: {len(test_signals)} signals")
    
    # Analyze single signal example
    print("\n2. Analyzing single signal with complete TDA pipeline...")
    
    example_signal = train_signals[0]
    example_label = y_train[0]
    
    # Extract features and intermediates
    features, intermediates = analyzer.analyze_signal(
        example_signal, return_intermediates=True
    )
    
    print(f"Example signal label: {'Arrhythmia' if example_label else 'Normal'}")
    print(f"Latest death time feature: {features['latest_death_time']:.4f}")
    print(f"H0 feature count: {features['H0_count']}")
    print(f"H0 max persistence: {features['H0_max_persistence']:.4f}")
    
    # Visualize the analysis
    print("\n3. Creating visualizations...")
    
    # Plot complete pipeline results
    fig = visualizer.plot_pipeline_results(
        signal=intermediates['processed_signal'],
        embedded=intermediates['embedded'],
        diagrams=intermediates['diagrams'],
        features=features,
        title=f"TDA Analysis - {'Arrhythmia' if example_label else 'Normal'} Signal"
    )
    
    plt.savefig('tda_pipeline_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved pipeline visualization to 'tda_pipeline_analysis.png'")
    
    # Plot mutual information analysis
    from tda_arrhythmia.core.embedding import TakensEmbedding
    embedder = TakensEmbedding()
    
    fig_mi = visualizer.plot_mutual_information(
        example_signal, 
        title="Optimal Time Delay Selection"
    )
    plt.savefig('mutual_information_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved mutual information plot to 'mutual_information_analysis.png'")
    
    # Train classifier
    print("\n4. Training TDA-based classifier...")
    
    # Train the analyzer
    analyzer.fit(train_signals, y_train)
    
    # Make predictions on test set
    print("\n5. Evaluating on test set...")
    
    test_predictions = []
    test_probabilities = []
    
    for signal in test_signals:
        pred, prob = analyzer.predict(signal, return_proba=True)
        test_predictions.append(pred)
        test_probabilities.append(prob)
    
    test_predictions = np.array(test_predictions)
    test_probabilities = np.array(test_probabilities)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"\nTest Accuracy: {accuracy:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_predictions, 
                               target_names=['Normal', 'Arrhythmia']))
    
    # Visualize classification results
    fig_results = visualizer.plot_classification_results(
        y_test, test_predictions, test_probabilities,
        title="TDA-based Arrhythmia Classification Results"
    )
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    print("Saved classification results to 'classification_results.png'")
    
    # Extract key feature for simple classification
    print("\n6. Demonstrating key 'latest death time' feature...")
    
    latest_death_features = []
    for signal in all_signals:
        latest_death = analyzer.extract_key_feature(signal)
        latest_death_features.append(latest_death)
    
    latest_death_features = np.array(latest_death_features)
    
    # Simple threshold classifier on this single feature
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(labels, latest_death_features)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    simple_predictions = (latest_death_features > optimal_threshold).astype(int)
    simple_accuracy = accuracy_score(labels, simple_predictions)
    
    print(f"Latest death time feature alone:")
    print(f"  AUC: {roc_auc:.3f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Accuracy with threshold: {simple_accuracy:.3f}")
    
    # Feature comparison visualization
    normal_indices = labels == 0
    arrhythmia_indices = labels == 1
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(latest_death_features[normal_indices], bins=20, alpha=0.7, 
             label='Normal', color='lightblue')
    plt.hist(latest_death_features[arrhythmia_indices], bins=20, alpha=0.7,
             label='Arrhythmia', color='salmon')
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Threshold = {optimal_threshold:.3f}')
    plt.xlabel('Latest Death Time')
    plt.ylabel('Frequency')
    plt.title('Latest Death Time Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Latest Death Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latest_death_time_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved latest death time analysis to 'latest_death_time_analysis.png'")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Complete TDA Pipeline Accuracy: {accuracy:.3f}")
    print(f"Single Feature (Latest Death Time) Accuracy: {simple_accuracy:.3f}")
    print(f"Latest Death Time AUC: {roc_auc:.3f}")
    print("\nThis demonstrates the power of TDA features for cardiac")
    print("arrhythmia detection, with the 'latest death time' being")
    print("a particularly discriminative topological feature.")
    
    # Show robustness analysis
    print("\n7. Robustness Analysis...")
    
    from tda_arrhythmia.utils.noise_handling import RobustTDAAnalyzer
    
    robust_analyzer = RobustTDAAnalyzer(n_bootstrap=20)  # Reduced for demo
    
    # Analyze one signal with confidence intervals
    robust_results = robust_analyzer.robust_analysis(example_signal)
    
    print("\nRobust Analysis Results (with 95% confidence intervals):")
    key_features = ['latest_death_time', 'H0_count', 'H0_max_persistence']
    
    for feat in key_features:
        if feat in robust_results:
            stats = robust_results[feat]
            print(f"{feat}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  95% CI: [{stats['lower']:.4f}, {stats['upper']:.4f}]")
            print(f"  Std: {stats['std']:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run example
    main()
    
    # Show plots if in interactive mode
    try:
        plt.show()
    except:
        pass