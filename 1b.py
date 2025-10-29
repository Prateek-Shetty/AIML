# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Load Dataset
df = pd.read_csv("housing_prices_SLR.csv", delimiter=',')
print("Data successfully loaded!")
print(df.head())

# Step 3: Feature Matrix (X) and Target Vector (y)
X = df[['AREA']].values       # Feature Matrix (2D)
y = df['PRICE'].values        # Target Vector (1D)

print("\nSample X values:", X[:5].flatten())
print("Sample y values:", y[:5])

# Step 4: Split the Data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Train the Simple Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("\nModel Parameters (with intercept):")
print(f"Intercept (b0): {lr_model.intercept_}")
print(f"Slope (b1): {lr_model.coef_[0]}")

# Model without intercept
lr_no_intercept = LinearRegression(fit_intercept=False)
lr_no_intercept.fit(X_train, y_train)
print("\nModel Parameters (without intercept):")
print(f"Intercept (b0): {lr_no_intercept.intercept_}")
print(f"Slope (b1): {lr_no_intercept.coef_[0]}")

# Step 6: Predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Step 7: R² Score Calculation
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print("\nModel Performance:")
print(f"R2 Score (Train): {r2_train}")
print(f"R2 Score (Test): {r2_test}")
print(f"R2 Score (via .score() method): {lr_model.score(X_test, y_test)}")

# Step 8: Visualization
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.scatter(X_test, y_test, color='blue', label='Testing Data')
plt.plot(X_train, y_train_pred, color='yellow', label='Regression Line')
plt.xlabel("Area of the House")
plt.ylabel("House Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.show()


# Comparing the training and testing R² score values,
# the accuracy of the simple linear regression model with respect to this dataset is average.
