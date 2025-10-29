# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB

# Step 2: Load Dataset
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, :-1]  # Drop last column if not needed
print("Shape of dataset:", df.shape)

# Step 3: Define Features and Target
X = df.iloc[:, 2:].values
y = df['diagnosis'].values
print("\nSample features (X):")
print(X[:5])
print("\nSample labels (y):")
print(y[:5])

# Step 4: Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=500
)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Baseline Model (always predicting 'B')
baseline_pred = ["B"] * len(y_train)
baseline_acc = accuracy_score(y_train, baseline_pred)
print("\nBaseline Model Accuracy:", baseline_acc)
print("\nBaseline Confusion Matrix:\n", confusion_matrix(y_train, baseline_pred))
print("\nBaseline Classification Report:\n", classification_report(y_train, baseline_pred))

# Step 6: Train Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Step 7: Evaluate Model
print("\nTraining Accuracy:", nb_model.score(X_train, y_train))
print("Testing Accuracy:", nb_model.score(X_test, y_test))

print("\nTraining Confusion Matrix:\n", confusion_matrix(y_train, nb_model.predict(X_train)))
print("\nTesting Confusion Matrix:\n", confusion_matrix(y_test, nb_model.predict(X_test)))

print("\nTraining Classification Report:\n", classification_report(y_train, nb_model.predict(X_train)))
print("\nTesting Classification Report:\n", classification_report(y_test, nb_model.predict(X_test)))

# Step 8: Probability Predictions
np.set_printoptions(suppress=True)
print("\nSample Prediction Probabilities (Train):")
print(nb_model.predict_proba(X_train)[:5])
print("\nSample Predicted Labels (Train):")
print(nb_model.predict(X_train)[:5])

# Step 9: Adjusting Probability Threshold
train_pred_threshold = nb_model.predict_proba(X_train)[:, 1] > 0.25
test_pred_threshold = nb_model.predict_proba(X_test)[:, 1] > 0.25

print("\nConfusion Matrix (Train, Threshold=0.25):")
print(confusion_matrix(y_train == 'M', train_pred_threshold))

print("\nConfusion Matrix (Test, Threshold=0.25):")
print(confusion_matrix(y_test == 'M', test_pred_threshold))
