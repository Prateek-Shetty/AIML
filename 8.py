# Import required libraries
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ------------------------------
# Parameters
# ------------------------------
max_features = 10000  # Number of words to consider as features (vocabulary size)
maxlen = 500          # Cut texts after this number of words (max input length)
batch_size = 32

# ------------------------------
# Load IMDB dataset
# ------------------------------
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

# ------------------------------
# Pad sequences so that all have same length
# ------------------------------
print('Padding sequences (samples x time)...')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# ------------------------------
# Define the RNN Model
# ------------------------------
model = Sequential()
model.add(Embedding(max_features, 32))   # Embedding layer (learns word vectors)
model.add(SimpleRNN(32))                 # Simple recurrent layer
model.add(Dense(1, activation='sigmoid'))# Binary output (positive or negative sentiment)

model.summary()

# ------------------------------
# Compile the model
# ------------------------------
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# Train the model
# ------------------------------
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# ------------------------------
# Evaluate the model
# ------------------------------
test_loss, test_acc = model.evaluate(input_test, y_test, verbose=1)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# ------------------------------
# Predict on test set
# ------------------------------
predicted_classes = (model.predict(input_test) > 0.5).astype("int32").flatten()

# ------------------------------
# Calculate correct / incorrect predictions
# ------------------------------
correct = np.where(predicted_classes == y_test)[0]
incorrect = np.where(predicted_classes != y_test)[0]
print(f"Found {len(correct)} correct labels")
print(f"Found {len(incorrect)} incorrect labels")

# ------------------------------
# Classification report
# ------------------------------
num_classes = 2
target_names = ["Negative (0)", "Positive (1)"]
print(classification_report(y_test, predicted_classes, target_names=target_names))

# ------------------------------
# Plot Training and Validation Accuracy
# ------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10,5))
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# ------------------------------
# Plot Training and Validation Loss
# ------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,5))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
