
"""Apply: 
Simple linear regression model for head Brain dataset and predict brain weight based  on head size using the least square method. 
Find out 
 i. R2score for the predicted model. 
ii. Display all the data points along with the fitting the data points to the 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# STEP 1: Load the dataset
# ==========================
data = pd.read_csv('headbrain.csv')
print("Data successfully loaded!")
print("Shape of data:", data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head())

# ==========================
# STEP 2: Extract variables
# ==========================
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
print("\nExtracted Features:")
print("X (Head Size) sample:", X[:5])
print("Y (Brain Weight) sample:", Y[:5])

# ==========================
# STEP 3: Compute means
# ==========================
mean_x = np.mean(X)
mean_y = np.mean(Y)
print("\nMean of X (Head Size):", mean_x)
print("Mean of Y (Brain Weight):", mean_y)

# ==========================
# STEP 4: Compute coefficients
# ==========================
n = len(X)
numer = np.sum((X - mean_x) * (Y - mean_y))
denom = np.sum((X - mean_x) ** 2)
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

print("\nRegression Coefficients:")
print(f"Slope (b1): {b1}")
print(f"Intercept (b0): {b0}")

# ==========================
# STEP 5: Plot regression line
# ==========================
max_x = np.max(X) + 100
min_x = np.min(X) - 100
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='#ef5423', label='Data Points')
plt.plot(x, y, color='#58b970', label='Regression Line')
plt.xlabel('Head Size (cm³)')
plt.ylabel('Brain Weight (grams)')
plt.title('Head Size vs Brain Weight Regression')
plt.legend()
plt.show()

# ==========================
# STEP 6: Compute R² Score
# ==========================
y_pred = b0 + b1 * X
ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - mean_y) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\nModel Evaluation:")
print(f"R2 Score: {r2}")

print("\nLinear Regression completed successfully!")

# The simple linear regression model gives average accuracy depending on the R² score value.
