# steeringThesisCodes
Codes for analyzing drowsiness through KSS, DRT and SWA data. Includes visualization, regression, correlations, feature extraction, entropy-based measures, and machine learning models for drowsiness detection.

# Drowsiness Detection Analysis

This repository contains four codes designed to investigate the relationship between **subjective sleepiness**, measured with the **Karolinska Sleepiness Scale (KSS)**, **objective cognitive performance**, measured with the **Detection Response Task (DRT)**, and **objective driver behavior**, mensured with the **Steering Wheel Angle (SWA)**.  

---

## 📌 Repository Structure
- **Code 1**:  
  Loads a CSV dataset, processes user/condition labels, and generates visualizations.  
  - Polynomial regression between KSS and DRT  
  - Boxplots of DRT across KSS levels  
  - Histograms of DRT and KSS distributions  
  - Time-series regressions (KSS vs. time, DRT vs. time)

- **Code 2**:  
  Performs regression and correlation analyses.  
  - Regression for KSS–DRT, KSS–time, DRT–time  
  - Pearson, Spearman, Distance Correlation, and Mutual Information  
  - ANOVA and t-tests  
  - Principal Component Analysis (PCA)  
  - KMeans clustering for user grouping

- **Code 3**:  
  Implements a **windowed feature extraction pipeline** applied to SWA signals with KSS labels.  
  - Generates statistical descriptors (skewness, kurtosis, std, peak-to-peak)  
  - Computes entropy-based features (traditional, approximate entropy, sample entropy)  
  - Produces a feature table with binary drowsy/alert labels  
  - Includes simple data augmentation for drowsy windows

- **Code 4**:  
  Benchmarks classical machine learning models for drowsiness detection.  
  - Reads extracted features from Code 3  
  - Evaluates classifiers: Logistic Regression, Gaussian NB, Decision Tree, Random Forest, SVM  
  - 5-fold cross-validation  
  - Performance metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, Pearson correlation  
  - Aggregates results for reporting
  
