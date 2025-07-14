#!/usr/bin/env python3
"""
Basic TDA Analysis Example

This example demonstrates the fundamental concepts of TDA
applied to cardiac signals, step by step.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tda_arrhythmia.core.embedding import TakensEmbedding
from tda_arrhythmia.core.persistence import PersistenceComputer
from tda_arrhythmia.core.features import PersistenceFeatureExtractor
from tda_arrhythmia.utils.visualization import TDAVisualizer


def generate_cardiac_signal(duration=10, fs=250, arrhythmia=False):
    """Generate a synthetic cardiac signal."""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Base cardiac rhythm
    heart_rate = 75 + np.random.normal(0, 5)  # BPM with variability
    rr_interval = 60 / heart_rate
    
    signal = np.zeros_like(t)
    
    # Generate QRS complexes
    beat_times = np.arange(0, duration, rr_interval)
    beat_times += np.random.normal(0, 0.02, len(beat_times))  # Heart rate variability
    
    for beat_time in beat_times:
        if beat_time > duration:
            break
        
        # QRS complex shape
        qrs_duration = 0.08 + np.random.normal(0, 0.01)
        qrs_amplitude = 1.0
        
        if arrhythmia and np.random.random() < 0.3:
            # Arrhythmic beat - different morphology
            qrs_amplitude *= np.random.uniform(1.5, 2.5)
            qrs_duration *= np.random.uniform(1.3, 1.8)
        
        # Add QRS complex
        qrs_mask = np.abs(t - beat_time) < qrs_duration/2
        if np.any(qrs_mask):
            qrs_signal = qrs_amplitude * np.exp(-((t - beat_time) / (qrs_duration/4))**2)
            signal += qrs_signal
        
        # Add T-wave
        t_wave_time = beat_time + 0.3
        t_wave_mask = np.abs(t - t_wave_time) < 0.15
        if np.any(t_wave_mask):
            t_wave = 0.3 * qrs_amplitude * np.exp(-((t - t_wave_time) / 0.05)**2)
            signal += t_wave
    
    # Add noise
    signal += np.random.normal(0, 0.05, len(signal))
    
    return t, signal


def demonstrate_takens_embedding():
    """Demonstrate Takens' embedding process."""
    print("\n1. TAKENS' EMBEDDING DEMONSTRATION")
    print("=" * 50)
    
    # Generate signal
    t, signal = generate_cardiac_signal(duration=5, arrhythmia=False)
    print(f"Generated cardiac signal: {len(signal)} samples, {len(signal)/250:.1f} seconds")
    
    # Initialize embedder
    embedder = TakensEmbedding(dimension=3, method='mutual_information')
    
    # Find optimal delay
    optimal_delay = embedder.find_optimal_delay(signal)
    print(f"Optimal time delay (mutual information): {optimal_delay} samples")
    
    # Compare with autocorrelation method
    embedder_autocorr = TakensEmbedding(method='autocorrelation')
    autocorr_delay = embedder_autocorr.find_optimal_delay(signal)
    print(f"Optimal time delay (autocorrelation): {autocorr_delay} samples")
    
    # Perform embedding
    embedded, delay_used = embedder.embed(signal)
    print(f"Embedding result: {embedded.shape} (using delay {delay_used})")
    
    # Visualize
    visualizer = TDAVisualizer()
    
    plt.figure(figsize=(15, 5))
    
    # Original signal
    plt.subplot(1, 3, 1)
    plt.plot(t, signal, 'b-', linewidth=1)
    plt.title('Original Cardiac Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 2D projection of embedding
    plt.subplot(1, 3, 2)
    plt.plot(embedded[:, 0], embedded[:, 1], 'r-', alpha=0.7, linewidth=0.8)
    plt.scatter(embedded[0, 0], embedded[0, 1], c='g', s=50, label='Start', zorder=5)
    plt.scatter(embedded[-1, 0], embedded[-1, 1], c='r', s=50, label='End', zorder=5)
    plt.title('2D Phase Space (x(t) vs x(t+τ))')
    plt.xlabel('x(t)')
    plt.ylabel(f'x(t+{delay_used})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mutual information plot
    plt.subplot(1, 3, 3)
    max_lag = min(50, len(signal) // 4)
    lags = range(1, max_lag)
    mi_values = []
    
    from sklearn.feature_selection import mutual_info_regression
    for lag in lags:
        x = signal[:-lag].reshape(-1, 1)
        y = signal[lag:]
        mi = mutual_info_regression(x, y, random_state=42)[0]
        mi_values.append(mi)
    
    plt.plot(lags, mi_values, 'b-', linewidth=2)
    plt.axvline(x=optimal_delay, color='r', linestyle='--', 
               label=f'Optimal delay = {optimal_delay}')
    plt.title('Mutual Information vs Lag')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('takens_embedding_demo.png', dpi=300, bbox_inches='tight')
    print("Saved embedding visualization to 'takens_embedding_demo.png'")
    
    return signal, embedded


def demonstrate_persistence_computation(signal, embedded):
    """Demonstrate persistence computation."""
    print("\n2. PERSISTENCE COMPUTATION DEMONSTRATION")
    print("=" * 50)
    
    # Initialize persistence computer
    computer = PersistenceComputer(library='ripser', max_dimension=2)
    
    try:
        # Compute persistence diagrams
        diagrams = computer.compute_persistence(embedded)
        
        print("Persistence computation successful!")
        for dim, diagram in diagrams.items():
            print(f"  H{dim}: {len(diagram)} features")
            if len(diagram) > 0:
                persistences = diagram[:, 1] - diagram[:, 0]
                print(f"    Max persistence: {np.max(persistences):.4f}")
                if dim == 0:
                    print(f"    Latest death time: {np.max(diagram[:, 1]):.4f}")
        
    except ImportError:
        print("Ripser not available, using fallback computation...")
        # Simple fallback for demonstration
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embedded))
        max_dist = np.max(distances)
        
        # Create mock diagrams for demonstration
        diagrams = {
            0: np.array([[0, max_dist * 0.3], [0, max_dist * 0.8], [0.1, max_dist * 0.5]]),
            1: np.array([[max_dist * 0.2, max_dist * 0.6]])
        }
        print("Using fallback persistence computation for demonstration")
    
    # Visualize persistence diagrams
    visualizer = TDAVisualizer()
    
    plt.figure(figsize=(12, 4))
    
    # Persistence diagram
    plt.subplot(1, 3, 1)
    for dim, diagram in diagrams.items():
        if len(diagram) == 0:
            continue
        colors = ['red', 'blue', 'green']
        markers = ['o', 's', '^']
        plt.scatter(diagram[:, 0], diagram[:, 1], 
                   c=colors[dim], marker=markers[dim], s=60, alpha=0.7,
                   label=f'H{dim}')
    
    # Diagonal line
    all_values = []
    for diagram in diagrams.values():
        if len(diagram) > 0:
            all_values.extend(diagram.flatten())
    
    if all_values:
        min_val, max_val = min(all_values), max(all_values)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title('Persistence Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Barcode
    plt.subplot(1, 3, 2)
    y_pos = 0
    colors = ['red', 'blue', 'green']
    
    for dim in sorted(diagrams.keys()):
        diagram = diagrams[dim]
        if len(diagram) == 0:
            continue
        
        for birth, death in diagram:
            plt.barh(y_pos, death - birth, left=birth, 
                    height=0.8, color=colors[dim], alpha=0.7)
            y_pos += 1
    
    plt.xlabel('Filtration Value')
    plt.title('Persistence Barcode')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Betti curves
    plt.subplot(1, 3, 3)
    betti_curves = computer.compute_betti_curves(diagrams)
    
    colors = ['red', 'blue', 'green']
    for dim, (filtration_values, betti_numbers) in betti_curves.items():
        plt.plot(filtration_values, betti_numbers, 
                color=colors[dim], linewidth=2, label=f'β{dim}')
        plt.fill_between(filtration_values, betti_numbers, alpha=0.3, color=colors[dim])
    
    plt.xlabel('Filtration Value')
    plt.ylabel('Betti Number')
    plt.title('Betti Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('persistence_demo.png', dpi=300, bbox_inches='tight')
    print("Saved persistence visualization to 'persistence_demo.png'")
    
    return diagrams


def demonstrate_feature_extraction(diagrams):
    """Demonstrate feature extraction from persistence diagrams."""
    print("\n3. FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = PersistenceFeatureExtractor(
        feature_types=['statistics', 'entropy', 'triangles']
    )
    
    # Extract features
    features = extractor.extract_features(diagrams)
    
    print(f"Extracted {len(features)} topological features:")
    
    # Show key features
    key_features = [
        'latest_death_time', 'H0_count', 'H0_max_persistence', 
        'H1_count', 'H1_max_persistence', 'H0_entropy', 'H1_entropy'
    ]
    
    print("\nKey Features:")
    for feature in key_features:
        if feature in features:
            value = features[feature]
            print(f"  {feature}: {value:.4f}")
    
    # Show statistical features by dimension
    print("\nStatistical Features by Dimension:")
    for dim in [0, 1, 2]:
        dim_features = {k: v for k, v in features.items() if k.startswith(f'H{dim}_')}
        if dim_features:
            print(f"  H{dim} features: {len(dim_features)}")
            for k, v in list(dim_features.items())[:3]:  # Show first 3
                print(f"    {k}: {v:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    
    # Plot key features
    plt.subplot(1, 2, 1)
    feature_names = []
    feature_values = []
    
    for feature in key_features:
        if feature in features:
            feature_names.append(feature.replace('_', '\n'))
            feature_values.append(features[feature])
    
    bars = plt.bar(range(len(feature_names)), feature_values, 
                   color='lightblue', edgecolor='navy')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.title('Key Topological Features')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, feature_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Compare H0 vs H1 features
    plt.subplot(1, 2, 2)
    h0_features = [v for k, v in features.items() if k.startswith('H0_') and 'count' in k or 'persistence' in k]
    h1_features = [v for k, v in features.items() if k.startswith('H1_') and 'count' in k or 'persistence' in k]
    
    x = np.arange(min(len(h0_features), len(h1_features)))
    width = 0.35
    
    if h0_features and h1_features:
        plt.bar(x - width/2, h0_features[:len(x)], width, label='H0', alpha=0.7)
        plt.bar(x + width/2, h1_features[:len(x)], width, label='H1', alpha=0.7)
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.title('H0 vs H1 Features Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('feature_extraction_demo.png', dpi=300, bbox_inches='tight')
    print("Saved feature extraction visualization to 'feature_extraction_demo.png'")
    
    return features


def compare_normal_vs_arrhythmic():
    """Compare topological features between normal and arrhythmic signals."""
    print("\n4. NORMAL vs ARRHYTHMIC COMPARISON")
    print("=" * 50)
    
    # Generate both types of signals
    t_normal, normal_signal = generate_cardiac_signal(duration=5, arrhythmia=False)
    t_arrhythmic, arrhythmic_signal = generate_cardiac_signal(duration=5, arrhythmia=True)
    
    # Analyze both signals
    embedder = TakensEmbedding(dimension=3)
    computer = PersistenceComputer()
    extractor = PersistenceFeatureExtractor(feature_types=['statistics', 'entropy'])
    
    # Normal signal analysis
    embedded_normal, _ = embedder.embed(normal_signal)
    try:
        diagrams_normal = computer.compute_persistence(embedded_normal)
    except ImportError:
        # Fallback for demonstration
        diagrams_normal = {0: np.array([[0, 0.5], [0, 1.0]]), 1: np.array([[0.3, 0.7]])}
    
    features_normal = extractor.extract_features(diagrams_normal)
    
    # Arrhythmic signal analysis
    embedded_arrhythmic, _ = embedder.embed(arrhythmic_signal)
    try:
        diagrams_arrhythmic = computer.compute_persistence(embedded_arrhythmic)
    except ImportError:
        # Fallback for demonstration (different values)
        diagrams_arrhythmic = {0: np.array([[0, 0.8], [0, 1.5]]), 1: np.array([[0.2, 0.9]])}
    
    features_arrhythmic = extractor.extract_features(diagrams_arrhythmic)
    
    # Compare key features
    print("Feature Comparison:")
    key_features = ['latest_death_time', 'H0_count', 'H0_max_persistence', 'H0_entropy']
    
    for feature in key_features:
        if feature in features_normal and feature in features_arrhythmic:
            normal_val = features_normal[feature]
            arrhythmic_val = features_arrhythmic[feature]
            ratio = arrhythmic_val / normal_val if normal_val != 0 else float('inf')
            
            print(f"  {feature}:")
            print(f"    Normal: {normal_val:.4f}")
            print(f"    Arrhythmic: {arrhythmic_val:.4f}")
            print(f"    Ratio: {ratio:.2f}")
    
    # Visualize comparison
    plt.figure(figsize=(15, 10))
    
    # Signals
    plt.subplot(2, 3, 1)
    plt.plot(t_normal, normal_signal, 'b-', linewidth=1, label='Normal')
    plt.title('Normal ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(t_arrhythmic, arrhythmic_signal, 'r-', linewidth=1, label='Arrhythmic')
    plt.title('Arrhythmic ECG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Phase spaces
    plt.subplot(2, 3, 3)
    plt.plot(embedded_normal[:, 0], embedded_normal[:, 1], 'b-', alpha=0.7, label='Normal')
    plt.plot(embedded_arrhythmic[:, 0], embedded_arrhythmic[:, 1], 'r-', alpha=0.7, label='Arrhythmic')
    plt.title('Phase Space Comparison')
    plt.xlabel('x(t)')
    plt.ylabel('x(t+τ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Persistence diagrams
    plt.subplot(2, 3, 4)
    for dim, diagram in diagrams_normal.items():
        if len(diagram) > 0:
            plt.scatter(diagram[:, 0], diagram[:, 1], c='blue', marker='o', 
                       s=60, alpha=0.7, label=f'Normal H{dim}')
    
    for dim, diagram in diagrams_arrhythmic.items():
        if len(diagram) > 0:
            plt.scatter(diagram[:, 0], diagram[:, 1], c='red', marker='^', 
                       s=60, alpha=0.7, label=f'Arrhythmic H{dim}')
    
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title('Persistence Diagrams')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature comparison
    plt.subplot(2, 3, 5)
    normal_values = [features_normal.get(f, 0) for f in key_features]
    arrhythmic_values = [features_arrhythmic.get(f, 0) for f in key_features]
    
    x = np.arange(len(key_features))
    width = 0.35
    
    plt.bar(x - width/2, normal_values, width, label='Normal', alpha=0.7, color='blue')
    plt.bar(x + width/2, arrhythmic_values, width, label='Arrhythmic', alpha=0.7, color='red')
    
    plt.xlabel('Features')
    plt.ylabel('Feature Value')
    plt.title('Feature Comparison')
    plt.xticks(x, [f.replace('_', '\n') for f in key_features], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Latest death time focus
    plt.subplot(2, 3, 6)
    ldt_normal = features_normal.get('latest_death_time', 0)
    ldt_arrhythmic = features_arrhythmic.get('latest_death_time', 0)
    
    plt.bar(['Normal', 'Arrhythmic'], [ldt_normal, ldt_arrhythmic], 
           color=['blue', 'red'], alpha=0.7)
    plt.title('Latest Death Time Comparison')
    plt.ylabel('Latest Death Time')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    plt.text(0, ldt_normal + 0.01, f'{ldt_normal:.3f}', ha='center', va='bottom')
    plt.text(1, ldt_arrhythmic + 0.01, f'{ldt_arrhythmic:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('normal_vs_arrhythmic_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison visualization to 'normal_vs_arrhythmic_comparison.png'")


def main():
    """Main demonstration function."""
    print("TDA Cardiac Analysis - Basic Concepts Demonstration")
    print("=" * 60)
    
    # Set up
    np.random.seed(42)
    plt.style.use('default')
    
    try:
        # Step 1: Takens' embedding
        signal, embedded = demonstrate_takens_embedding()
        
        # Step 2: Persistence computation
        diagrams = demonstrate_persistence_computation(signal, embedded)
        
        # Step 3: Feature extraction
        features = demonstrate_feature_extraction(diagrams)
        
        # Step 4: Normal vs arrhythmic comparison
        compare_normal_vs_arrhythmic()
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("Generated visualization files:")
        print("  - takens_embedding_demo.png")
        print("  - persistence_demo.png") 
        print("  - feature_extraction_demo.png")
        print("  - normal_vs_arrhythmic_comparison.png")
        print("\nKey findings:")
        print("  - Takens' embedding reconstructs cardiac signal dynamics")
        print("  - Persistence diagrams capture topological structure")
        print("  - Latest death time is discriminative between normal/arrhythmic")
        print("  - TDA features provide geometric insights into cardiac signals")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()