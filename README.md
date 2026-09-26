🫀 ECG AI Benchmark: Deep Learning & Machine Learning Benchmarking for Electrocardiogram Classification
	An end-to-end benchmark framework designed to evaluate, compare, and validate deep learning and machine learning models for automated Electrocardiogram (ECG) heartbeat categorization and arrhythmia detection.
	
📌 Table of Contents
1.Key Innovations & Novel Features
2.Approach & Technical Rationale
3.Datasets Used
4.Project Architecture
5.Environment Setup & Installation
6.Running the Benchmark
7.Expected Outputs & Verification
8.Visual Demonstrations
9.Evaluation Metrics & Results
10.License

💡 Key Innovations & Novel Features

This benchmark framework moves beyond traditional black-box classification by introducing technical features designed specifically for physiological signal processing:

1. Multi-Domain Hybrid Feature Fusion

	Time + Frequency + Morphological Synergy: Rather than relying solely on raw waveform inputs, our models utilize a dual-stream fusion approach. Deep 1D convolutional layers extract raw morphology (QRS complex dimensions, ST-segment deviations), while parallel branches incorporate Discrete Wavelet Transforms (DWT) and Heart Rate Variability (HRV) frequency metrics.

	Attention-Guided Beat Weighting: An integrated temporal attention mechanism dynamically focuses on clinically critical segments (such as subtle T-wave inversions and P-wave absences) while attenuating motion artifact intervals.

2. Zero-Leakage Patient-Stratified Cross-Validation

	Inter-Patient Evaluation Standard: Unlike conventional benchmarks that perform random train-test splitting (which suffers from severe data leakage by placing beats from the same patient in both splits), our pipeline enforces strict inter-patient splitting.

	Models are evaluated strictly on unseen patient cohorts, reflecting true clinical real-world generalization.

3. Adaptive Noise-Robust Preprocessing Pipeline

	Dual-Stage Filtering: Implements an adaptive Butterworth bandpass filter coupled with Median Baseline Correction to eliminate baseline wander without distorting the sensitive low-frequency ST segment.

	Dynamic R-Peak Centering: Employs an automated peak-detection alignment algorithm that dynamically centers heartbeats even in the presence of premature contractions (PVCs) and tachycardia.

4. Edge-Centric Efficiency & Quantization Benchmark

	Evaluates models not only on diagnostic accuracy (F1-score) but also on clinical edge viability:

	Floating-point operations (FLOPs) and memory footprint.

	Inference latency on CPU and embedded hardware profiles.

	INT8 Post-Training Quantization (PTQ) readiness for wearable deployment.

🔬 Approach & Technical Rationale

1. Methodology Overview

Electrocardiogram signals are non-stationary, quasi-periodic biomedical time-series vulnerable to motion artifacts, power-line interference, and baseline wandering. This project implements a standardized benchmarking pipeline:

Signal Preprocessing & Denoising:

Bandpass filtering (0.5 Hz – 45 Hz Butterworth filter) removes baseline wander and high-frequency EMG noise.

Z-score / Min-Max normalization standardizes signal amplitudes across varying patient recordings.

Beat segmentation / Fixed-window extraction standardizes lead inputs.

Model Benchmarking:

1D Convolutional Neural Networks (1D-CNN): Selected for localized temporal feature extraction (QRS complexes, P/T waves) with low computational footprint.

Recurrent / Hybrid Architectures (CNN-LSTM / GRU / Transformer): Captures long-range temporal dependencies and rhythm irregularities across multi-lead sequences.

Baseline ML (Random Forest / XGBoost with feature extraction): Provides an interpretable, resource-efficient benchmark using morphological and Heart Rate Variability (HRV) metrics.

2. Why This Approach?

Clinical Relevance: Raw ECG processing must handle inter-patient variability. Comparative benchmarking shows which architectures preserve diagnostic fidelity without overfitting to specific recording devices.

Reproducibility: A unified evaluation pipeline ensures that every model is scored against the exact same test fold and cross-validation splits.

Inference Efficiency: Compares parameter counts and latency to identify which architectures are viable for deployment on edge monitoring devices.

📊 Datasets Used

This project supports standard open-access clinical ECG benchmarks:

Dataset

Description

Source / Link

PTB-XL ECG Dataset

Large-scale 12-lead clinical dataset containing 21,837 clinical ECG records from 18,885 patients.

PhysioNet PTB-XL

MIT-BIH Arrhythmia Database

Standard 2-channel ambulatory ECG recordings across 48 half-hour excerpts annotated for beat classifications.

PhysioNet MIT-BIH

Kaggle ECG Heartbeat Dataset

Curated CSV subsets derived from MIT-BIH & PTB Diagnostic databases (AAMI EC57 standard).

Kaggle Dataset Link
