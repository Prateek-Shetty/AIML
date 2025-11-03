"""Multiple linear regression model for housing_prices dataset and predict house price  based on the area, floor and room size of the house using the library scikit learn. Find out  the accuracy of the model using R2score statistics for the predicted model. 
"""


# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Step 2: Load Dataset
df = pd.read_csv("housing_prices.csv")
df = df.iloc[:, [0, 1, 2, 4]]  # Selecting required columns
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Feature Matrix and Target Vector
X = df.iloc[:, :3].values
y = df.iloc[:, 3].values
print("\nSample feature values (X):")
print(X[:5])
print("\nSample target values (y):")
print(y[:5])

# Step 4: Split Data (80% Train, 20% Test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Train the Multiple Linear Regression Model

mlr_model = LinearRegression(fit_intercept=True)
mlr_model.fit(X_train, y_train)

# Step 6: Display Model Parameters
print("\nModel Parameters:")
print("Intercept (b0):", mlr_model.intercept_)
print("Coefficients (b1, b2, b3):", mlr_model.coef_)

# Step 7: Evaluate Model Performance
print("\nModel Performance:")
print("R2 Score (Train):", mlr_model.score(X_train, y_train))
print("R2 Score (Test):", mlr_model.score(X_test, y_test))


#The multiple linear regression model accuracy is good with respect to this  dataset by comparing R2 training and testing score values.