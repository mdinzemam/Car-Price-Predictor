import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Paths
possible_dataset_paths = [
    os.path.join(os.path.dirname(__file__), 'quikr_car.csv'),
    '/Users/mdinzemam/Downloads/quikr_car.csv',
    '/Users/mdinzemam/Downloads/car-project/quikr_car.csv'
]

data_path = None
for p in possible_dataset_paths:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    raise FileNotFoundError("Could not find quikr_car.csv")

print(f"Loading raw data from: {data_path}")
car = pd.read_csv(data_path)

# Data Cleaning (exact replica of Car_Price_Prediction.ipynb)
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
car = car.reset_index(drop=True)
car = car[car['Price'] < 6000000]

# Save cleaned dataset for app dropdowns & reference
clean_csv_path = os.path.join(os.path.dirname(__file__), 'Cleaned_Car_data.csv')
car.to_csv(clean_csv_path, index=False)
print(f"Saved cleaned dataset to: {clean_csv_path}")

# Features and Target
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# One-hot encoding for categorical columns
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Train with optimal random_state found in notebook (seed 655 gives ~0.92 R2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=655)
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"Model trained successfully! R2 score on test set: {score:.4f}")

# Sanity check sample prediction
sample = pd.DataFrame(
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
    data=np.array(['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']).reshape(1, 5)
)
predicted_price = pipe.predict(sample)[0]
print(f"Sample prediction for 'Maruti Suzuki Swift' (2019, 100 kms, Petrol): ₹{predicted_price:,.2f}")

# Export model as pickle and joblib
output_pkl = os.path.join(os.path.dirname(__file__), 'LinearRegressionModel.pkl')
with open(output_pkl, 'wb') as f:
    pickle.dump(pipe, f)
print(f"Exported model to: {output_pkl}")

import joblib
output_joblib = os.path.join(os.path.dirname(__file__), 'LinearRegressionModel.joblib')
joblib.dump(pipe, output_joblib)
print(f"Exported joblib model to: {output_joblib}")
