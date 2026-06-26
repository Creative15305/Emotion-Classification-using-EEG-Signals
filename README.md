# Emotion Classification Using EEG Signals

A machine learning project that classifies human emotions from EEG brain signals using the DEAP dataset. Emotions are mapped to four quadrants of the valence–arousal model. The pipeline covers traditional ML classifiers as well as deep learning models (CNN, ResNet50, DenseNet121), with real-time emotion output via Firebase.

---

## Overview

EEG (Electroencephalography) provides a direct, non-invasive measure of brain activity. This project builds an end-to-end framework to:
- Preprocess raw EEG signals and extract spectrograms
- Classify emotions using both classical ML and deep learning models
- Push real-time emotion predictions to Firebase

---

## Dataset

**DEAP Dataset** — A Database for Emotion Analysis using Physiological Signals
- 32 participants, each exposed to 40 emotionally stimulating music video clips
- EEG recordings with self-reported valence, arousal, and dominance ratings
- [Dataset link](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

> The dataset is not included in this repo. Download it separately and update the `DATA_DIR` path in the notebook.

---

## Methodology

### 1. Preprocessing
- Downsampling to **128 Hz**
- Artifact removal (eye blinks, muscle movements)
- Normalization and segmentation
- Labeling based on valence–arousal ratings

### 2. Feature Extraction
Two approaches used:

**A. Welch's Power Spectral Density (PSD)**
Features extracted across four EEG frequency bands:
| Band | Frequency Range |
|------|----------------|
| Theta | 4–8 Hz |
| Alpha | 8–12 Hz |
| Beta | 12–30 Hz |
| Gamma | 30–45 Hz |

**B. Spectrogram Generation**
- EEG signals converted to 2D spectrograms using `scipy.signal.spectrogram`
- Log-scaled to reduce dynamic range
- Averaged across all 32 EEG channels per trial

### 3. Emotion Mapping (Valence–Arousal Model)
| Quadrant | Emotion |
|----------|---------|
| High Valence + High Arousal | Happy |
| High Valence + Low Arousal | Excited |
| Low Valence + High Arousal | Angry |
| Low Valence + Low Arousal | Sad |

### 4. Models

**Classical ML (on PSD features):**
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

**Deep Learning (on spectrograms):**
- Custom CNN (2D Conv layers)
- ResNet50 (transfer learning, fine-tuned last 10 layers)
- DenseNet121 (feature extraction) + PCA (64 components) + SVM (4-class)

### 5. Real-Time Firebase Integration
- Predicted valence and arousal fed into emotion mapping
- Final emotion label pushed to Firebase Realtime Database
- Enables live display on connected hardware/dashboard

---

## Results

| Task | Best Classifier | Accuracy |
|------|----------------|----------|
| Valence Classification | KNN | **62.33%** |
| Arousal Classification | MLP | **58.75%** |

> Accuracies in the 55–65% range are typical for EEG-based emotion classification due to inter-subject variability and signal noise — consistent with published literature.

---

## Tech Stack

- **Python**, **Google Colab**
- **Scikit-learn** — KNN, SVM, MLP, PCA
- **TensorFlow/Keras** — CNN, ResNet50
- **PyTorch + torchvision** — DenseNet121 feature extraction
- **SciPy** — Welch's PSD, spectrogram generation
- **NumPy / Pandas / Matplotlib / Seaborn** — data handling and visualization
- **Firebase Admin SDK** — real-time emotion output
- **joblib** — model persistence

---

## Project Structure

```
├── EEG.ipynb                   # Main notebook (full pipeline)
├── data/                       # DEAP dataset 
└── README.md
```
