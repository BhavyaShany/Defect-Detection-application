# Project Specification: Real-Time Defect Detection & Classification System

## Problem Description
Design a real-time defect detection and classification system for a complex manufacturing process, specifically focusing on identifying anomalies in surface roughness of cast metal parts. The system must integrate Computer Vision and Machine Learning techniques to analyze high-resolution 3D scans of the parts.

## Objectives
* **Understand and Apply ML & CV**: Understand the application of Machine Learning and Computer Vision in industrial defect detection.
* **Efficient Pipeline Design**: Design an efficient data acquisition and preprocessing pipeline for handling 3D scans (PLY, PCD, STL, OBJ) and optical images.
* **CNN-Based Real-Time Classification**: Implement real-time classification using Convolutional Neural Networks (CNNs).
* **Performance Evaluation**: Evaluate the system's performance using metrics such as precision, recall, and F1-score.

## Proposed Solution
The system comprises:
1. **High-Resolution 3D Scanning System & Parser**: Captures surface details and ingests multiple 3D point cloud and mesh representations.
2. **Python-Based Data Processing & Feature Extraction Module**: Calculates surface roughness, normal vectors, and PCA/ICP pose alignment to extract 2D depth and normal maps.
3. **TensorFlow-Based CNN Classifier**: Real-time defect detection and classification engine providing instant pass/fail/review verdicts.
4. **Integration with Manufacturing Execution System (MES)**: Automated quality control, transactional outbox dispatching, and dynamic quality alert generation.

## Technology Stack
* **Programming & Core**: Python, OpenCV, NumPy, SciPy, Open3D
* **Machine Learning & Deep Learning**: TensorFlow, PyTorch, Keras, Scikit-Learn
* **Backend & API**: FastAPI, Uvicorn, WebSockets, SQLAlchemy (SQLite / PostgreSQL)
* **Frontend & Visualization**: React 18, Vite, Three.js (interactive 3D point cloud & mesh viewer)

