# Artificial Dataset Classification

A binary classification pipeline designed for an artificial dataset with 100 numerical input features. The primary optimization and evaluation metric is **Balanced Accuracy**.

The central modelling challenge is isolating informative structure from noise. The final approach pairs feature scaling, tree-based feature selection, and SMOTE with a distance-weighted $k$-Nearest Neighbors ($k$-NN) classifier, achieving approximately **0.90 Balanced Accuracy** in cross-validation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Characteristics](#dataset-characteristics)
- [Model Screening](#model-screening)
- [Final Pipeline Architecture](#final-pipeline-architecture)
- [Validation Strategy](#validation-strategy)
- [Results Summary](#results-summary)
- [Inference / Test Predictions](#inference--test-predictions)
- [Summary of Experiments](#summary-of-experiments)

---

## Dataset Characteristics

- **Total samples:** 1,500 observations (training set)
- **Features:** 100 numerical input features
- **Missing values:** None
- **Target classes:** 2 (binary classification)

### Class Distribution

| Class | Observations | Proportion |
| :---: | :---: | :---: |
| **1** | 755 | ~50.3% |
| **2** | 745 | ~49.7% |

*The dataset is nearly balanced. Initial Pearson correlation analysis showed no strong individual linear relationships between single features and the target, highlighting the need for models capable of capturing nonlinear or local decision boundaries.*

---

## Model Screening

Multiple model families were evaluated during initial exploration:

- **Linear & Discriminant Models:** Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA). These lagged behind tree- and distance-based alternatives.
- **Tree-Based Ensembles:** Random Forest and Gradient Boosting showed strong, competitive baselines. Random Forest was subsequently adopted for feature selection.
- **Distance-Based Models:** A $k$-NN pipeline preceded by normalization and feature selection delivered the strongest results.

### Baseline Comparison

| Pipeline | Balanced Accuracy |
| :--- | :---: |
| **$k$-NN (with scaling & feature selection)** | **~0.9000** |
| **Random Forest** | 0.8674 |
| **Gradient Boosting** | 0.8440 |

---

## Final Pipeline Architecture

```text
Input Features (100)
       │
       ▼
 1. StandardScaler
       │
       ▼
 2. Random Forest Feature Selection (SelectFromModel)
       │
       ▼
 3. SMOTE
       │
       ▼
 4. Distance-Weighted k-NN (weights="distance")
       │
       ▼
 Predicted Class Probabilities
```

### Pipeline Components

1. **Feature Scaling (`StandardScaler`)**
   Standardization prevents features with naturally wider ranges from dominating Euclidean distance metrics in $k$-NN.
2. **Feature Selection (`SelectFromModel` with Random Forest)**
   Removes noisy and uninformative variables from the original 100 features. The selection threshold was set to:
   $$\text{Threshold} = 1.4 \times \text{mean feature importance}$$
3. **SMOTE (Synthetic Minority Over-sampling Technique)**
   Although the target classes are well-balanced, incorporating SMOTE consistently improved validation stability and downstream performance across tested preprocessing variations.
4. **Distance-Weighted $k$-NN (`weights="distance"`)**
   Closer neighbors exert higher influence over classification than farther neighbors, preserving local structure within the reduced feature space.

---

## Validation Strategy

Performance was verified using **Repeated Stratified $K$-Fold Cross-Validation**:

- **Folds:** 5
- **Repetitions:** 20
- **Total validation iterations:** $5 \times 20 = 100$ runs

Stratification ensures consistent class proportions across splits, while repetitions mitigate variance stemming from any single train-validation split.

---

## Results Summary

- **Mean Balanced Accuracy:** **~0.90**
- **Observed Fold Range:** **0.86 – 0.93**

Scores consistently stabilized around 0.90. Further increases in model complexity or more aggressive dimension reduction did not yield systematic gains.

---

## Inference / Test Predictions

The final pipeline was applied to generate class-probability predictions for **500 test observations**. 

Predicting soft probabilities rather than discrete labels preserves confidence scores, supporting downstream metric evaluation and potential threshold optimization.

---

## Summary of Experiments

Throughout the modelling process, the following configurations were evaluated:

- Baseline classifiers: Logistic Regression, LDA, QDA, Random Forest, Gradient Boosting, $k$-NN
- Alternative feature selection thresholds and estimators
- Preprocessing combinations (scaling variants, SMOTE vs. no oversampling)
- Distance-weighted vs. uniform weighting schemes for $k$-NN

**Conclusion:** For this artificial dataset, the most successful strategy is pruning noisy dimensions via tree importance prior to fitting a localized, distance-weighted classifier.
