# Speech Emotion Recognition with Wav2Vec2

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Wav2Vec2](https://img.shields.io/badge/Wav2Vec2-Speech_AI-success)
![IEMOCAP](https://img.shields.io/badge/Dataset-IEMOCAP-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A transformer-based Speech Emotion Recognition (SER) system using **Wav2Vec2**, **Attentive Statistical Pooling**, and **Additive Margin Softmax Loss** to classify emotions from speech recordings on the IEMOCAP dataset.

---

# Project Overview

Speech Emotion Recognition (SER) enables machines to identify human emotions from speech signals and has applications in:

- Human-computer interaction
- Contact center analytics
- Mental health assessment
- Voice assistants
- Conversational AI

This project explores multiple deep learning architectures before developing an enhanced Wav2Vec2-based model capable of significantly improving emotion classification performance on the IEMOCAP dataset.

---

# Project Highlights

✅ Transformer-based Speech Emotion Recognition

✅ Transfer Learning using Facebook Wav2Vec2

✅ Attentive Statistical Pooling (ASP)

✅ Additive Margin Softmax (AMS) Loss

✅ Extensive Architecture Comparison

✅ Speaker-independent Evaluation

✅ Deep Learning Experiment Tracking

---

# Dataset

The project uses the **IEMOCAP (Interactive Emotional Dyadic Motion Capture)** dataset.

Dataset characteristics:

- Approximately 12 hours of emotional speech
- 10 professional actors
- 4 emotion classes

  - Happy
  - Angry
  - Sad
  - Neutral

- Approximately 5,500 utterances
- Speaker-independent train/test split

---

# Model Evolution

Multiple deep learning architectures were evaluated throughout the project.

| Model | Performance |
|--------|------------|
| CRNN | 55–62% |
| CNN Variants | ~63.9% |
| Vision Transformer | ~62.8% |
| Wav2Vec2 | ~64% |
| Wav2Vec2 + ASP | ~66.7% |
| Wav2Vec2 + AMS | ~66.2% |
| Wav2Vec2 + ASP + AMS | **74.42%** |

The project demonstrates a systematic progression from traditional CNN-based architectures to transformer-based speech models enhanced with specialized pooling and loss functions.

---

# System Architecture

<img width="872" height="261" alt="image" src="https://github.com/user-attachments/assets/1c10e855-6186-4da9-8ec1-7c827fbc0e05" />


The final system consists of:

- Audio preprocessing
- Wav2Vec2 feature extraction
- Attentive Statistical Pooling
- Fully connected classification head
- Additive Margin Softmax Loss
- Four-class emotion prediction

---

# Feature Representation

Several feature extraction approaches were explored:

- Log Mel Spectrograms
- Wavelet Features
- Raw Audio Waveforms

<img width="738" height="394" alt="image" src="https://github.com/user-attachments/assets/b0e68fcf-4444-4476-a6be-d5390cc04a62" />


The final model leveraged raw waveforms with a pretrained Wav2Vec2 backbone to learn task-specific speech representations.

---

# Training Methodology

The training pipeline incorporated several optimization techniques:

- Layer-specific learning rates
- AdamW optimizer
- Mixed precision training
- Linear warmup scheduler
- ReduceLROnPlateau scheduler
- Early stopping
- Balanced sampling
- Stratified cross-validation

---

# Results

The final architecture exceeded traditional Speech Emotion Recognition benchmarks on the IEMOCAP dataset.

Key improvements included:

- Higher classification accuracy
- Faster convergence
- Improved class separation
- Better generalization
- Reduced confusion between similar emotions

<img width="822" height="316" alt="image" src="https://github.com/user-attachments/assets/81130e4f-6952-4f91-bd9c-d687a7c0da68" />


---

# Confusion Matrix

The confusion matrix demonstrates improved separation between emotionally similar classes, particularly reducing confusion between **Sad** and **Neutral** speech.

<img width="484" height="412" alt="image" src="https://github.com/user-attachments/assets/f9127249-331a-4631-b572-5d31ad625d71" />


---

# Technology Stack

## Programming

- Python

## Deep Learning

- PyTorch
- Hugging Face Transformers
- Wav2Vec2

## Machine Learning

- Scikit-learn
- NumPy
- Pandas

## Audio Processing

- Librosa
- Torchaudio

## Visualization

- Matplotlib
- Seaborn

## Experiment Tracking

- MLflow

---

# Repository Structure

```text
Speech-Emotion-Recognition-with-Wav2Vec2/

├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── speech_emotion_recognition_wav2vec2.ipynb
│
└── images/
    ├── architecture.png
    ├── performance_progression.png
    ├── confusion_matrix.png
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/AShirsat96/Speech-Emotion-Recognition-with-Wav2Vec2.git
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Download the IEMOCAP dataset from the official source and configure the dataset path inside the notebook before training.

---

# Skills Demonstrated

This project demonstrates practical experience with:

- Speech Emotion Recognition
- Transformer-based Deep Learning
- Transfer Learning
- Audio Signal Processing
- Wav2Vec2
- Feature Engineering
- Hyperparameter Optimization
- Model Evaluation
- Deep Learning Experimentation
- MLflow Experiment Tracking

---

# Project Context

This project was completed as part of a graduate **Deep Learning** course in a team of three.

### My Contributions

- Dataset preprocessing and preparation
- Balanced sampling strategy
- Baseline CRNN implementation
- Spectrogram feature engineering
- MLflow experiment tracking
- Contributions to the methodology and experimental evaluation

The repository documents the complete project while acknowledging the collaborative nature of its development.

---

# Future Improvements

Potential future enhancements include:

- Multimodal emotion recognition
- Real-time speech emotion prediction
- Model compression for edge deployment
- Data augmentation using SpecAugment
- Multi-task learning
- Cross-language emotion recognition
- Model ensemble techniques

---

# About the Author

**Aniket Shirsat**

AI Engineer | Data Scientist | Generative AI

🌐 Portfolio  
https://aniketdshirsat.com

💼 LinkedIn  
https://www.linkedin.com/in/aniketshirsatsg/

📧 Email  
aniketshirsat@hotmail.com

---

# License

This project is licensed under the MIT License.
