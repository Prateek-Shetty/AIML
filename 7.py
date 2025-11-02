"""Using Keras and tensor flow network 
i) Load the mnist image dataset 
ii) Design a two-layer neural network with one hidden layer and one output layer a. Use CNN with Leaky Relu activation function for the hidden layer b. Use sigmoid activation function for the output layer 
iii)Train the designed network for mnist dataset 
iv)Visualize the results of 
a) Training vs validation accuracy 
b) Training vs Validation loss 
"""


# Importing required libraries
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load MNIST dataset
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Find unique digits and their count
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Display sample training and test images
plt.figure(figsize=[5,5])

plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

# Reshape to include channel dimension (for CNN)
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# Normalize pixel values (0–255 → 0–1)
train_X = train_X.astype('float32') / 255
test_X = test_X.astype('float32') / 255

# Convert labels to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# Split training data into train and validation
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

batch_size = 64
epochs = 3
num_classes = 10

# ================================
# First CNN Model (Without Dropout)
# ================================
m_model = Sequential()
m_model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(28,28,1), padding='same'))
m_model.add(LeakyReLU(alpha=0.1))
m_model.add(MaxPooling2D((2,2), padding='same'))
m_model.add(Flatten())
m_model.add(Dense(128, activation='linear'))
m_model.add(LeakyReLU(alpha=0.1))
m_model.add(Dense(num_classes, activation='softmax'))

m_model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

m_model.summary()

m_train = m_model.fit(train_X, train_label,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(valid_X, valid_label))

# Evaluate the model
test_eval = m_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Plot accuracy and loss
accuracy = m_train.history['accuracy']
val_accuracy = m_train.history['val_accuracy']
loss = m_train.history['loss']
val_loss = m_train.history['val_loss']
epochs_range = range(len(accuracy))

plt.plot(epochs_range, accuracy, '--', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, '--', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# =====================================
# CNN Model with Dropout (to reduce overfitting)
# =====================================
m_model = Sequential()
m_model.add(Conv2D(32, kernel_size=(3,3), activation='linear', padding='same', input_shape=(28,28,1)))
m_model.add(LeakyReLU(alpha=0.1))
m_model.add(MaxPooling2D((2,2), padding='same'))
m_model.add(Dropout(0.25))

m_model.add(Flatten())
m_model.add(Dense(128, activation='linear'))
m_model.add(LeakyReLU(alpha=0.1))
m_model.add(Dropout(0.3))
m_model.add(Dense(num_classes, activation='softmax'))

m_model.summary()
m_model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

m_train_dropout = m_model.fit(train_X, train_label,
                              batch_size=batch_size,
                              epochs=3,
                              verbose=1,
                              validation_data=(valid_X, valid_label))

# Save the model
m_model.save("mnist_model_dropout.h5")

# Evaluate dropout model
test_eval = m_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Plot accuracy and loss (Dropout Model)
accuracy = m_train_dropout.history['accuracy']
val_accuracy = m_train_dropout.history['val_accuracy']
loss = m_train_dropout.history['loss']
val_loss = m_train_dropout.history['val_loss']
epochs_range = range(len(accuracy))

plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy (with Dropout)')
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss (with Dropout)')
plt.legend()
plt.show()

# =====================================
# Display Correctly and Incorrectly Classified Images
# =====================================
predicted_classes = m_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

correct = np.where(predicted_classes == test_Y)[0]
print("Found %d correct labels" % len(correct))

plt.figure(figsize=(10,10))
for i, correct_idx in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct_idx].reshape(28,28), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct_idx], test_Y[correct_idx]))
plt.tight_layout()
plt.show()

incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))

plt.figure(figsize=(10,10))
for i, incorrect_idx in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect_idx].reshape(28,28), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect_idx], test_Y[incorrect_idx]))
plt.tight_layout()
plt.show()

# Classification report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


"""Conclusion: Using Keras and tensor flow network loaded the mnist image dataset and designed a two-layer neural network with one hidden layer and one output layer using  CNN with Leaky Relu activation function for the hidden layer."""