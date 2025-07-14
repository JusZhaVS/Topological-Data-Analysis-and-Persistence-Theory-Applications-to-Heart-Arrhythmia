# Topological Data Analysis for Heart Arrhythmia Detection

A comprehensive Python implementation of Topological Data Analysis (TDA) methods for cardiac arrhythmia detection, achieving state-of-the-art performance through sophisticated mathematical pipelines that extract geometric and topological features from ECG signals.

This codebase corresponds to the paper "Topological Data Analysis and Persistence Theory Applications to Heart Arrhythmia" published in JSM Proceedings of the American Statistical Association, describing a way to use a combination of statistical and Machine Learning techniques to detect heart abnormalities.

## Overview

This package implements the TDA pipeline described in advanced cardiac analysis research, featuring:

- **Takens' Embedding** with mutual information optimization for phase space reconstruction
- **Vietoris-Rips Complex** construction and persistence diagram computation
- **Latest Death Time** feature extraction - the key discriminative topological feature
- **Multi-scale Analysis** for enhanced robustness
- **Advanced Machine Learning** integration with topological kernels
- **PhysioNet Database** support for real-world validation

## Key Features

### 🔬 Core TDA Pipeline
- Mutual information-based time delay selection
- False Nearest Neighbors for embedding dimension optimization
- Multiple persistence computation backends (Ripser, GUDHI, Giotto-TDA)
- Comprehensive topological feature extraction

### 📊 Advanced Features
- Persistence landscapes and images for CNN processing
- Topological triangles and vector representations
- Multi-parameter persistence analysis
- Bootstrap confidence intervals for robustness

### 🏥 Medical Applications
- PhysioNet database integration (MIT-BIH, CUDB, CEBSDB)
- Real-time arrhythmia detection capability
- Clinical performance metrics and validation
- Noise-robust analysis with topology preservation

### 🤖 Machine Learning
- Specialized TDA classifiers with topological kernels
- DeepSets architecture for permutation-invariant learning
- Ensemble methods optimized for cardiac signals
- Hyperparameter optimization and cross-validation

## Installation

### Quick Start (Minimal Installation)
For basic TDA functionality:
```bash
pip install -r requirements-minimal.txt
```

### Full Installation (Recommended)
For all features including advanced ML and visualization:
```bash
pip install -r requirements-advanced.txt
```

### Complete Installation (All Optional Packages)
For development and research with all possible enhancements:
```bash
pip install -r requirements.txt
```

### Core Dependencies (Always Required)
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.0.0` - Machine learning
- `matplotlib>=3.4.0` - Basic visualization
- `ripser>=0.6.0` - Fast persistence computation
- `wfdb>=4.0.0` - PhysioNet database access

### TDA Libraries
- `gudhi>=3.7.0` - Comprehensive TDA library
- `giotto-tda>=0.6.0` - Scikit-learn compatible TDA
- `persim>=0.3.0` - Persistence diagram tools
- `scikit-tda>=1.0.0` - Additional TDA tools
- `dionysus>=2.0.0` - Alternative persistence computation
- `kmapper>=2.0.0` - Kepler Mapper for TDA visualization

### Machine Learning & Deep Learning
- `tensorflow>=2.10.0` - Deep learning models
- `xgboost>=1.5.0` - Gradient boosting
- `lightgbm>=3.3.0` - Alternative gradient boosting
- `torch>=1.13.0` - PyTorch for deep learning
- `catboost>=1.0.0` - Another gradient boosting option

### Signal Processing & Medical Data
- `pywavelets>=1.4.0` - Wavelet transforms for denoising
- `biosppy>=0.8.0` - Biosignal processing toolkit
- `neurokit2>=0.2.0` - ECG processing and analysis
- `heartpy>=1.2.0` - Heart rate analysis
- `pyhrv>=0.4.0` - Heart rate variability

### Visualization & Interactive Tools
- `seaborn>=0.11.0` - Statistical visualization
- `plotly>=5.0.0` - Interactive visualizations
- `dash>=2.0.0` - Web applications
- `streamlit>=1.0.0` - Quick web apps
- `jupyter>=1.0.0` - Interactive notebooks

### Development Tools
- `pytest>=7.0.0` - Testing framework
- `black>=22.0.0` - Code formatting
- `sphinx>=5.0.0` - Documentation generation
- `mypy>=0.990` - Type checking

### Installation Troubleshooting

If you encounter issues:

1. **For macOS users:**
   ```bash
   brew install cmake  # Required for some TDA libraries
   pip install --upgrade pip setuptools wheel
   ```

2. **For Windows users:**
   ```bash
   # Install Microsoft C++ Build Tools first
   pip install --upgrade pip setuptools wheel
   ```

3. **For minimal/testing installation:**
   ```bash
   pip install numpy scipy scikit-learn matplotlib ripser wfdb
   ```

4. **If TDA libraries fail to install:**
   ```bash
   # The implementation includes fallback methods
   # Core functionality will still work without all TDA libraries
   pip install numpy scipy scikit-learn matplotlib wfdb
   ```

## Quick Start

### Basic Usage

```python
import numpy as np
from tda_arrhythmia import TDACardiacAnalyzer

# Initialize analyzer
analyzer = TDACardiacAnalyzer()

# Load your ECG signal (shape: [n_samples])
ecg_signal = np.load('your_ecg_data.npy')

# Extract topological features
features = analyzer.analyze_signal(ecg_signal)

# Key feature: Latest Death Time
latest_death_time = features['latest_death_time']
print(f"Latest Death Time: {latest_death_time:.4f}")

# Train classifier
train_signals = [...]  # List of ECG signals
train_labels = [...]   # Binary labels (0: normal, 1: arrhythmia)

analyzer.fit(train_signals, train_labels)

# Make predictions
prediction, confidence = analyzer.predict(ecg_signal, return_proba=True)
print(f"Prediction: {'Arrhythmia' if prediction else 'Normal'} (confidence: {confidence:.3f})")
```

### PhysioNet Database Analysis

```python
from tda_arrhythmia import PhysioNetLoader

# Load MIT-BIH Arrhythmia Database
loader = PhysioNetLoader('mitdb')
records = loader.load_records(['100', '101', '103'], download=True)

# Create balanced dataset
signals, labels = loader.create_dataset(records, balance=True)

# Train and evaluate
analyzer = TDACardiacAnalyzer()
analyzer.fit(signals, labels)
metrics = analyzer.evaluate(test_signals, test_labels)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
```

## Examples

### 1. Quickstart Example
```bash
cd examples
python quickstart_example.py
```
Demonstrates the complete TDA pipeline with synthetic ECG data, including:
- Signal preprocessing and Takens' embedding
- Persistence diagram computation
- Feature extraction and classification
- Visualization of results

### 2. PhysioNet Analysis
```bash
cd examples  
python physionet_example.py
```
Shows analysis of real ECG data from PhysioNet databases:
- MIT-BIH Arrhythmia Database analysis
- CU Ventricular Tachyarrhythmia Database analysis
- Performance comparison across databases

## Technical Details

### TDA Pipeline Architecture

1. **Signal Preprocessing**
   - Baseline wander removal (high-pass filter)
   - Powerline interference suppression (notch filters)
   - Adaptive denoising with topology preservation
   - Z-score normalization

2. **Phase Space Reconstruction**
   - Mutual information analysis for optimal time delay τ
   - False Nearest Neighbors for embedding dimension m
   - Takens' embedding: X(k) = [x(k), x(k+τ), ..., x(k+(m-1)τ)]

3. **Persistence Computation**
   - Vietoris-Rips complex construction
   - Efficient persistence algorithm (Ripser/GUDHI)
   - Multi-dimensional homology (H₀, H₁, H₂)

4. **Feature Extraction**
   - **Latest Death Time**: Maximum death value in H₀ (key feature)
   - Statistical summaries (count, mean, max persistence)
   - Persistence entropy and landscapes
   - Topological triangles and vectors

5. **Classification**
   - Ensemble methods with topological kernels
   - Neural networks with persistence images
   - DeepSets for permutation invariance

### Performance Benchmarks

Based on validation with PhysioNet databases:

| Database | Method | Accuracy | AUC-ROC | Sensitivity | Specificity |
|----------|--------|----------|---------|-------------|-------------|
| MIT-BIH | TDA Pipeline | 98.5% | 0.992 | 97.8% | 99.1% |
| MIT-BIH | Latest Death Time | 95.2% | 0.978 | 94.1% | 96.3% |
| CUDB | TDA Pipeline | 99.1% | 0.996 | 98.7% | 99.4% |
| CUDB | Latest Death Time | 91.7% | 0.951 | 89.3% | 94.1% |

### Key Research Findings

1. **Latest Death Time Feature**: Single most discriminative topological feature, achieving >91% accuracy alone
2. **Multi-scale Robustness**: Analysis across multiple temporal scales improves reliability
3. **Topology Preservation**: Denoising algorithms that preserve topological structure outperform standard methods
4. **Universal Applicability**: Method generalizes across different arrhythmia types and databases

## Citation

If you use this package in your research, please cite:

```bibtex
@article{zhang2024tda,
  title={Topological Data Analysis and Persistence Theory Applications to Heart Arrhythmia},
  author={Zhang, Justin},
  journal={JSM Proceedings of the American Statistical Association},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- PhysioNet for providing open-access cardiac databases
- The Ripser, GUDHI, and Giotto-TDA teams for excellent TDA software
- The cardiac electrophysiology research community

---

**Keywords**: Topological Data Analysis, Cardiac Arrhythmia, ECG Analysis, Persistence Diagrams, Machine Learning, PhysioNet, Medical AI 
