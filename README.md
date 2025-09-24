# Multi-Class Brain Tumor Classification with MRI Scans

## Overview
This project addresses the critical challenge of brain tumor classification using MRI scans through both classical machine learning and deep learning approaches. It explores and compares various methodologies to automate and improve diagnostic accuracy for four tumor classes: glioma, meningioma, pituitary tumor, and no tumor.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Classical Machine Learning Approach](#classical-machine-learning-approach)
  - [Feature Extraction with HOG](#feature-extraction-with-hog)
  - [Dimensionality Reduction with PCA](#dimensionality-reduction-with-pca)
  - [Training and Evaluation](#training-and-evaluation)
- [Deep Learning Approach](#deep-learning-approach)
  - [CNN Architecture](#cnn-architecture)
  - [Training Procedure](#training-procedure)
- [Results and Comparison](#results-and-comparison)
- [Conclusion](#conclusion)
- [References](#references)

---

## Dataset
- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: `glioma`, `meningioma`, `pituitary`, and `no tumor`
- **Total Images**: 7023 (balanced across classes)

---

## Preprocessing
- All images resized to **224x224** pixels
- Applied data augmentation (random flips, rotations, brightness/contrast variation)
- Pixel values normalized
- Data split into:
  - **Training**: 70%
  - **Validation**: 15%
  - **Test**: 15%

---

## Classical Machine Learning Approach
### Feature Extraction with HOG
- Used **Histogram of Oriented Gradients (HOG)** to extract shape and texture features from images.

### Dimensionality Reduction with PCA
- Applied **Principal Component Analysis (PCA)** to reduce feature dimensionality while retaining most variance.

### Training and Evaluation
- Trained the following classifiers:
  - Logistic Regression
  - Linear Discriminant Analysis
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Ensemble Classifier (Voting with SVM, RF, KNN)
- Best accuracy: **94.05%** with Ensemble Classifier

---

## Deep Learning Approach
### CNN Architecture
- Custom CNN with:
  - 3 convolutional blocks (Conv2D + ReLU + MaxPooling)
  - Fully connected layers with Dropout (rate=0.5)
  - Final Softmax output for 4-class prediction

### Training Procedure
- Optimizer: `Adam`, Learning rate: `0.001`
- Loss Function: `CrossEntropyLoss`
- Trained for **30 epochs**
- Model checkpointing based on best validation accuracy

---

## Results and Comparison
| Model                  | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 87.19%   |
| SVM                   | 93.21%   |
| Ensemble Classifier   | 94.05%   |
| **CNN**               | **98.00%** |

- Confusion matrix confirms excellent performance, particularly on `pituitary` and `no tumor` classes
- CNN significantly outperforms all classical models in all metrics

---

## Conclusion
This project demonstrates the superior performance of deep learning (CNN) over classical methods for multi-class brain tumor classification from MRI scans. While classical models with HOG+PCA achieved strong results (up to 94.05%), CNN reached **98% accuracy**, showing high potential for clinical applications in automated diagnosis.

---

## References
1. [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

