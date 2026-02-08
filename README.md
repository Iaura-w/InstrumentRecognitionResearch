# Audio Instrument Recognition Research

A comprehensive research project focused on the automatic classification of musical instruments (Piano, Violin, Acoustic Guitar, Cello) using Machine Learning and Deep Learning techniques.

## Overview
This project, developed as part of a Master's Thesis, explores the effectiveness of various model architectures in recognizing timbre. By leveraging Digital Signal Processing (DSP) and modern ML frameworks, the system achieves high accuracy in identifying instruments from raw audio files.

## Project Components
This project is split into two specialized repositories:
1. **Research & Backend (This Repo):** Feature extraction (MFCC), model benchmarking, and Flask API deployment.
2. **Mobile Frontend:** [InstrumentFinder Android App](https://github.com/Iaura-w/InstrumentFinder) – A Kotlin-based application that interfaces with the ML model to provide real-time recognition.

## Tech Stack
* **Mathematics:** Signal Analysis, Statistical Modeling, MFCC extraction.
* **Backend & ML:** Python, Scikit-learn, TensorFlow/Keras, Flask.
* **Mobile:** Kotlin, Android SDK.
* **Libraries:** Scikit-learn, TensorFlow/Keras, Librosa, Flask, Joblib.

## Feature Engineering
The core of the project relies on **MFCC (Mel-frequency cepstral coefficients)**. 
- Raw audio signals from the IRMAS dataset were processed using `librosa`.
- Features were aggregated (mean) to create a robust 20-dimensional representation for each audio sample.
- Data was balanced and split into training/testing sets (80/20) with cross-validation.

## Experimental Results

The research compared classical machine learning models with neural networks. Below are the precise results obtained from the experiments.

### 1. Binary Classification (2 Instruments: Piano & Violin)
*Dataset: 1,301 samples.*

| Model | Cross-validation | Accuracy | Precision | F1-Score | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | **85.82%** | **85.82%** | **85.63%** | **85.66%** | **85.69%** |
| **GB** | 83.93% | 83.14% | 83.39% | 82.72% | 82.41% |
| **RFC** | 82.37% | 83.90% | 83.75% | 85.66% | 83.62% |
| **MP (MLP)** | 83.51% | 80.08% | 81.73% | 80.25% | 79.22% |
| **CNN** | 80.06% | 82.38% | 82.12% | 82.05% | 81.99% |

### 2. Multi-class Classification (4 Instruments)
*Dataset: 2,326 samples (Piano, Violin, Acoustic Guitar, Cello).*

| Model | Cross-validation | Accuracy | Precision | F1-Score | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GB** | **67.37%** | 65.88% | 65.32% | 64.59% | 64.24% |
| **SVM** | 67.16% | **67.17%** | 66.50% | **65.94%** | **65.62%** |
| **RFC** | 66.94% | 65.02% | 63.37% | 61.44% | 61.43% |
| **MP (MLP)** | 54.27% | 55.36% | **66.80%** | 62.91% | 59.67% |
| **CNN** | 60.11% | 54.72% | 56.49% | 52.96% | 54.88% |

## Conclusions
* **Best Performance:** For binary classification, **SVM** achieved the highest accuracy and cross-validation score (85.82%). For four classes, **Gradient Boosting (GB)** showed the best generalization with a 67.37% cross-validation score.
* **Classical vs. Deep Learning:** Classical models (SVM, GB, RFC) consistently outperformed neural networks (CNN, MP) on this specific dataset and feature set.
* **Reliability:** The results show high consistency between accuracy and cross-validation, indicating that the models generalize well to new audio data.

### Key Findings
* **Classical Superiority:** For this specific dataset size and MFCC feature set, classical models (SVM, GB) provided better stability and higher generalization scores (CV).
* **Neural Network Limitations:** CNN and MLP models showed signs of difficulty in identifying distinct instrumental timbres, likely due to the need for a larger dataset or more complex spectral input beyond mean MFCCs.
* **Robustness:** The high correlation between Accuracy and Cross-Validation scores in SVM/GB models confirms that the system is not overfitted and performs reliably on unseen audio samples.
* **Implementation:** The best-performing Gradient Boosting model was deployed via a Flask API to serve the Android application.

## Deployment
The project includes a production-ready **Flask API** (`app.py`) that:
1. Accepts `.wav` or `.mp3` files via POST requests.
2. Performs real-time preprocessing and MFCC extraction.
3. Returns a probability distribution for each instrument class.

*Note: The model is automatically fetched from a remote server upon deployment for efficiency.*

## Repository Structure
* `/notebooks` - Step-by-step experiments (data prep, MFCC, model training).
* `/src` - Core scripts including the Flask API.
* `requirements.txt` - Dependency list for easy environment setup.
* `/results_images` - Confusion Matrices images
