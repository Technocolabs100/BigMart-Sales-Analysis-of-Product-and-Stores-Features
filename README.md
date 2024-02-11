# BigMart-Sales-Analysis-of-Product-and-Stores-Features
This repository contains code for a sales prediction project using machine learning. The project involves data preprocessing, exploratory data analysis (EDA), and the application of various regression models for predicting item outlet sales. The models implemented include Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and XGBoost Regressor.

# Getting Started
To run the code, make sure you have Python installed along with the required libraries. You can install the necessary packages using the following command:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# Loading Packages & Data
The initial section of the code focuses on loading required packages and the training/test datasets for analysis.

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Load the training and test datasets
df_train = pd.read_csv("train.csv")

df_test = pd.read_csv("test.csv")

# Data Cleaning
The EDA section explores the dataset to understand its structure, missing values, and statistical properties.

df_train.info()

df_train.isnull().sum()

df_train["Item_Weight"].fillna(df_train["Item_Weight"].mean(), inplace=True)

df_train["Outlet_Size"].fillna(df_train["Outlet_Size"].mode()[0], inplace=True)

df_train.drop(["Item_Identifier", "Outlet_Identifier"], axis=1, inplace=True)

# Exploratory Data Analysis
import seaborn as sns

sns.heatmap(df_train.corr(), annot=True)

(various plots and visualizations)

# Model Preprocessing
The preprocessing section involves label encoding categorical features and standardizing numerical features.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_train = df_train.apply(le.fit_transform)

X = df_train.drop(columns="Item_Outlet_Sales", axis=1)

y = df_train["Item_Outlet_Sales"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

# 1) Linear Regression
The code includes the implementation of Linear Regression and evaluation metrics.

from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(X_train_std, y_train)

y_pred_lr = lr.predict(X_test_std)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

## evaluation metrics
r2_score: 0.5632512232226438

mean_absolute_error: 480.17010792231474

root_mean_squared_error: 599.5058290564374

# 2) Ridge Regression
Ridge Regression is implemented using grid search for hyperparameter tuning.

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

ridge_model = Ridge()

## Define hyperparameter grid for grid search
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

grid_ridge = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_ridge.fit(X_train_std, y_train)

print("Best alpha:", grid_ridge.best_params_['alpha'])

best_ridge = grid_ridge.best_estimator_

y_pred_ridge = best_ridge.predict(X_test_std)

## evaluation metrics
r2_score: 0.5632495934455479

mean_absolute_error: 480.17545575189223

root_mean_squared_error: 599.5069476170487

# 3) Lasso Regression
Similar to Ridge, Lasso Regression is implemented with hyperparameter tuning.

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso = Lasso()

## Define hyperparameter grid for grid search
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

grid = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')

grid.fit(X_train_std, y_train)

print("Best alpha:", grid.best_params_['alpha'])

best_lasso = grid.best_estimator_

y_pred_lasso = best_lasso.predict(X_test_std)

## evaluation metrics
r2_score: 0.5632663795399471

mean_absolute_error: 480.1601628708069

root_mean_squared_error: 599.4954267612457

# 4) Random Forest
Random Forest regression is implemented using grid search for hyperparameter tuning.

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()

## Define hyperparameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

rf_grid.fit(X_train, y_train)

best_params = rf_grid.best_params_

print("Best Parameters:", best_params)

best_rf_reg = rf_grid.best_estimator_

y_pred_rf = best_rf_reg.predict(X_test)

## evaluation metrics
r2_score: 0.6712587648844144

mean_absolute_error: 399.0869127762937

root_mean_squared_error: 520.1213160320432

# 5) XGBoost Regressor
XGBoost Regressor is implemented with grid search for hyperparameter tuning.

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

xgb = XGBRegressor()

## Define hyperparameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, verbose=1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best Parameters:", best_params)

best_xgb_reg = grid_search.best_estimator_

y_pred_xgb = best_xgb_reg.predict(X_test)

## evaluation metrics
r2_score: 0.674441170770061

mean_absolute_error: 396.1303481798368

root_mean_squared_error: 517.5976555021037


# Conclusion
From above 5 models, Random Forest and XGBoost regressor have high accuracy.



