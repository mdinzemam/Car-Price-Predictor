# 🚗 Used Car Price Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)

An end-to-end Supervised Machine Learning pipeline built to predict secondary market automobile valuations based on technical specifications and historical market metrics.

---

## 📌 Project Overview
Estimating vehicle resale prices requires analyzing non-linear correlations across mileage, vehicle age, fuel economy, transmission type, and brand tiers. This project implements data preprocessing, outlier mitigation, and regression algorithms to deliver valuation estimates.

---

## ⚙️ ML Pipeline Workflow

1. **Exploratory Data Analysis (EDA):**
   - Handled missing attributes and structural data types.
   - Identified and treated price and mileage outliers using Interquartile Range (IQR) filtering.
2. **Feature Engineering & Transformation:**
   - Standardized numerical features for model consistency.
   - Applied categorical encoding across transmission, fuel types, and vehicle brand categories.
3. **Model Training & Evaluation:**
   - Evaluated Linear Regression and Ensemble methods.
   - Assessed accuracy using **$R^2$ Score**, **MAE**, and **RMSE** metrics.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mdinzemam/Car-Price-Predictor.git](https://github.com/mdinzemam/Car-Price-Predictor.git)
   cd Car-Price-Predictor
