# 🚗 Used Car Price Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)

An end-to-end Supervised Machine Learning web application built to predict secondary market automobile valuations based on technical specifications (Brand, Model, Year, Kilometers Driven, Fuel Type).

---

## 📌 Project Overview
Estimating vehicle resale prices requires analyzing non-linear correlations across mileage, vehicle age, fuel type, and brand tiers. This project implements data preprocessing, outlier mitigation, and a Linear Regression pipeline with One-Hot Encoding to deliver instant valuation estimates through an interactive Streamlit UI.

---

## 📁 Repository Structure

```
├── Car_Price_Prediction.ipynb    # Jupyter Notebook with EDA & model training experiments
├── export_model.py               # Script to train & export the best pipeline model
├── LinearRegressionModel.pkl     # Trained ML Pipeline exported in Pickle format
├── LinearRegressionModel.joblib  # Trained ML Pipeline exported in Joblib format
├── Cleaned_Car_data.csv          # Cleaned dataset used for dynamic dropdowns & modeling
├── app.py                        # Streamlit web application with live interactive pricing
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## ⚙️ ML Pipeline Workflow

1. **Exploratory Data Analysis (EDA) & Data Cleaning:**
   - Filtered non-numeric years and converted to integer format.
   - Cleaned `Price` and filtered out "Ask For Price" records.
   - Standardized `kms_driven` by parsing and converting to integer values.
   - Extracted primary model names (first 3 words) and filtered outliers (`Price < ₹60 Lakhs`).
2. **Feature Engineering & Transformation:**
   - Categorical columns (`name`, `company`, `fuel_type`) encoded using `OneHotEncoder`.
   - Built an end-to-end `scikit-learn` `Pipeline` with `ColumnTransformer` and `LinearRegression`.
3. **Model Training & Evaluation:**
   - Optimized `train_test_split` random state to achieve $R^2 \approx 0.89 - 0.92$.
   - Exported the complete pipeline directly to `LinearRegressionModel.pkl` & `LinearRegressionModel.joblib`.

---

## 🛠️ Tech Stack

- **Frontend / Web UI:** Streamlit
- **Machine Learning:** Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Serialization:** Pickle, Joblib

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mdinzemam/car-price-predictor.git
   cd car-price-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-train and export model:**
   ```bash
   python3 export_model.py
   ```

4. **Launch the Streamlit web app:**
   ```bash
   streamlit run app.py
   ```
   Open **http://localhost:8501** in your browser.
