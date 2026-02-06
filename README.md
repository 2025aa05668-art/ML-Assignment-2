# Obesity Type Classification – Machine Learning Assignment 2

## a. Problem Statement

The objective of this project is to build multiple machine learning classification models to predict **obesity type** based on health, lifestyle, and demographic features. The goal is to compare different classification algorithms and deploy an interactive Streamlit application that allows users to upload test data, select models, and visualize evaluation results.

Target variable: **NObeyesdad (Obesity Type)**
Features used:
Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS.

---

## b. Dataset Description

The dataset used is an **Obesity Classification dataset** obtained from a public repository (UCI). It contains demographic, eating habit, and physical activity related attributes used to classify individuals into different obesity categories.

### Dataset Characteristics

* Multiclass classification problem
* Number of features: 16
* Instances: More than 2111 samples
* Includes both categorical and numerical attributes

### Target Classes

Typical obesity classes include:

* Insufficient Weight
* Normal Weight
* Overweight Level I
* Overweight Level II
* Obesity Type I
* Obesity Type II
* Obesity Type III

---

## c. Models Used and Evaluation Metrics

Six classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

### Evaluation Metrics Comparison

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.8723   | 0.9881 | 0.8769    | 0.8723 | 0.8699   | 0.8522 |
| Decision Tree            | 0.9362   | 0.9628 | 0.9367    | 0.9362 | 0.9362   | 0.9254 |
| KNN                      | 0.8203   | 0.9579 | 0.8118    | 0.8203 | 0.8020   | 0.7938 |
| Naive Bayes              | 0.5154   | 0.8911 | 0.5315    | 0.5154 | 0.4625   | 0.4473 |
| Random Forest (Ensemble) | 0.9409   | 0.9959 | 0.9431    | 0.9409 | 0.9414   | 0.9311 |
| XGBoost (Ensemble)       | 0.9551   | 0.9985 | 0.9564    | 0.9551 | 0.9550   | 0.9477 |

---

## Model Performance Observations

| ML Model            | Observation                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs well as a baseline model with good AUC but slightly lower accuracy compared to ensemble models.          |
| Decision Tree       | Provides strong performance and interpretability but may risk overfitting.                                        |
| KNN                 | Moderate performance; sensitive to feature scaling and data distribution.                                         |
| Naive Bayes         | Lowest performance likely due to strong independence assumptions among features.                                  |
| Random Forest       | Robust ensemble method with high accuracy and balanced metrics.                                                   |
| XGBoost             | Best overall performance with highest accuracy, AUC, and MCC due to boosting technique and better generalization. |

---

## Streamlit Application Features

* CSV dataset upload (test dataset)
* Model selection dropdown
* Automatic evaluation metrics display
* Confusion matrix visualization

---