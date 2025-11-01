"""Using Keras and Tensor flow framework 
i) Load the Pima_indians_diabetes dataset 
ii) Design a two-layer neural network with one hidden layer and one output layer a. Use Relu activation function for the hidden layer 
b. Use sigmoid activation function for the output layer 
iii) Train the designed network for Pima_indians_diabetes 
iv)Evaluate the network 
v) Generate Predictions for 10 samples 
"""



# First Neural Network with Keras

from numpy import loadtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras import layers




# ==========================
# Step 1: Load the dataset
# ==========================
dataframe = pd.read_csv('pima-indians-diabetes.csv', delimiter=',')
dataframe.head()

# ==========================
# Step 2: Split into input (X) and output (y)
# ==========================
X = dataframe.iloc[:, :8]   # First 8 columns as features
y = dataframe.iloc[:, 8]    # Last column as target
dataframe.shape

# Split data into training and test sets (67% train, 33% test)
features_train, features_test, target_train, target_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

# ==========================
# Step 3: Define the Keras model
# ==========================
network = models.Sequential()

# Input + first hidden layer
network.add(Dense(units=8, activation="relu", input_shape=(features_train.shape[1],)))

# Second hidden layer
network.add(Dense(units=8, activation="relu"))

# Output layer (sigmoid for binary classification)
network.add(Dense(units=1, activation="sigmoid"))

# ==========================
# Step 4: Compile the model
# ==========================
network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ==========================
# Step 5: Train the model
# ==========================
history = network.fit(
    features_train, target_train,
    epochs=20,
    verbose=1,
    batch_size=100,
    validation_data=(features_test, target_test)
)

# ==========================
# Step 6: Visualize Training and Test Loss
# ==========================
training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.show()

# ==========================
# Step 7: Evaluate the model on training data
# ==========================
_, accuracy = network.evaluate(features_train, target_train)
print('Training Accuracy: %.2f' % (accuracy * 100))

# ==========================
# Step 8: Make predictions
# ==========================
predicted_target = network.predict(features_test)

# Evaluate accuracy on test data
_, accuracy = network.evaluate(features_test, target_test)
print('Testing Accuracy: %.2f' % (accuracy * 100))

# Display first 10 predicted probabilities
for i in range(10):
    print(predicted_target[i])

# ==========================
# Step 9: Plot Accuracy Curves
# ==========================
training_accuracy = history.history["accuracy"]
test_accuracy = history.history["val_accuracy"]

plt.plot(epoch_count, training_accuracy, "r--")
plt.plot(epoch_count, test_accuracy, "b-")
plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.title("Training vs Testing Accuracy")
plt.show()



"""Conclusion :Using Keras and Tensor flow framework loaded the Pima_indians_diabetes  dataset and designed a two-layer neural network with one hidden layer and one output  layer and generated predictions for 10 samples. """