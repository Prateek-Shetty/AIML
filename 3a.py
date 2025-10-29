# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv('breast_cancer.csv')
print("Data successfully loaded!")
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Step 3: Select Features and Target
df = df.iloc[:, :-1]  # Drop last column if not needed
X = df.iloc[:, 2:].values   # Features
y = df['diagnosis'].values  # Target variable

print("\nSample feature values (X):")
print(X[:2])
print("\nSample target values (y):")
print(y[:5])

# Step 4: Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Initialize and Train the Model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = dt_classifier.predict(X_test)
prob_predictions = dt_classifier.predict_proba(X_test)

# Step 7: Evaluate the Model
print("\nModel Evaluation Results:")
print("Training Accuracy:", accuracy_score(y_train, dt_classifier.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, y_pred))

print("\nTraining Confusion Matrix:\n", confusion_matrix(y_train, dt_classifier.predict(X_train)))
print("\nTesting Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
