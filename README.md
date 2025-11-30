## Project Title

Credit Card Fraud Detection 

---

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset contains highly imbalanced classes, where fraudulent transactions represent only a tiny fraction of the total data. To address this challenge, the project explores data preprocessing, feature scaling, model building, and evaluation using metrics suitable for imbalanced datasets.

Two machine learning models were used:

Logistic Regression: A baseline linear model that provides quick interpretability and probabilistic outputs.

Random Forest Classifier: A powerful ensemble model capable of capturing non-linear patterns and improving prediction performance.

The performance of both models was evaluated primarily using AUPRC (Area Under the Precision-Recall Curve), which is more meaningful for imbalanced classification than accuracy. The Random Forest model produced higher recall and overall better discrimination, making it more effective for identifying fraudulent transactions while minimizing false negatives.

**This project demonstrates practical skills in model development, data analysis, and evaluation for real-world fraud detection workflows.**
---

## Features

The Credit Card Fraud Detection dataset consists of 30 input features and 1 target label.
Most features are anonymized using Principal Component Analysis (PCA) to protect confidentiality.

1.PCA-Transformed Features (V1–V28)

The features V1 to V28 are the result of PCA transformation applied to the original customer and transaction information.

PCA helps maintain privacy while reducing dimensionality and correlations.

2.Time

Indicates the number of seconds elapsed between each transaction and the first transaction in the dataset.

Useful for observing time-based transaction patterns.

3.Amount

Represents the monetary value of the transaction.

Can be used for cost-sensitive learning (higher loss for misclassifying high-value fraudulent transactions).

4.Class (Target Variable)

0 → Legitimate transaction

1 → Fraudulent transaction

The dataset is highly imbalanced (fraud cases are extremely rare).
---

## Tech Stack

-- Programming Language

Python

-- Libraries & Frameworks

NumPy – Numerical computations

Pandas – Data loading, cleaning, and manipulation

Matplotlib & Seaborn – Data visualization and plotting

Scikit-Learn (sklearn) – Machine Learning models and evaluation metrics

-Logistic Regression

-Random Forest Classifier

-Train-test split

-StandardScaler

-Precision-Recall & AUC metrics

--Development Environment

Google Colab / Jupyter Notebook

---

## 📂 Folder Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
|
└── README.md


```

---

## Installation & Setup

1. CLONE THE REPOSITORY
   https://github.com/kirticode09/credit_card_fraud_detection.git
   cd credit_card_fraud_detection

2. INSTALL REQUIRED LIBRARIES
   numpy
   pandas
   matplotlib
   seaborn
   scikit-learn

3. RUN THE NOTEBOOK

---

## Contributing

This is a personal academic project, so contributions are not required.

---

##  License

This project is licensed under the MIT License.

---
