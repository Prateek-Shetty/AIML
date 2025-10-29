"""Apply: 
a)Multiple linear regression model for student dataset and predict writing skill of  student based on the math skill and reading skill of the student using the Gradient  descent method. Find out R2score for the predicted model. """

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 2: Load Dataset
data = pd.read_csv('student.csv')
print("Data Shape:", data.shape)
print(data.head())

# Step 3: Extract Features and Target
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

# Step 4: 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(math, read, write, color='#ef1234')
ax.set_xlabel("Math")
ax.set_ylabel("Reading")
ax.set_zlabel("Writing")
ax.set_title("3D Scatter Plot of Student Scores")
plt.show()

# Step 5: Prepare Feature Matrix X and Target Vector Y
m = len(math)
x0 = np.ones(m)
X = np.array([x0, math, read]).T
Y = np.array(write)

# Step 6: Initialize Coefficients
B = np.array([0, 0, 0])
alpha = 0.0001

# Step 7: Define Cost Function
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
    return J

initial_cost = cost_function(X, Y, B)
print("\nInitial Cost:", initial_cost)

# Step 8: Gradient Descent Function
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

# Step 9: Run Gradient Descent
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

# Step 10: Output Results
print("\nNew Coefficients (B):", newB)
print("Final Cost:", cost_history[-1])

# Step 11: Predictions
Y_pred = X.dot(newB)

# Step 12: Evaluation Metrics
def rmse(Y, Y_pred):
    return np.sqrt(np.sum((Y - Y_pred) ** 2) / len(Y))

def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = np.sum((Y - mean_y) ** 2)
    ss_res = np.sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

print("\nRMSE:", rmse(Y, Y_pred))
print("R2 Score:", r2_score(Y, Y_pred))



#The accuracy of the multiple linear regression model is good depending on the R2score  value