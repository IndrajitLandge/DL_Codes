""" Image Classification on MNIST Using DNN with Learning Rate Variation
● Use the MNIST dataset and build a DNN
● Train the same model using learning rates: 0.01, 0.001
● Use SGD optimizer and track accuracy for each run
● Plot loss and accuracy for comparison
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to build DNN model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes for MNIST digits
    ])
    return model

# Train model with a given learning rate
def train_model(lr):
    model = create_model()
    model.compile(
        optimizer=SGD(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        verbose=0
    )
    return history

# Train with learning rate 0.01
history_01 = train_model(0.01)

# Train with learning rate 0.001
history_001 = train_model(0.001)

# Plot loss comparison
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(history_01.history['loss'], label='LR=0.01 Train Loss')
plt.plot(history_01.history['val_loss'], label='LR=0.01 Val Loss')
plt.plot(history_001.history['loss'], label='LR=0.001 Train Loss')
plt.plot(history_001.history['val_loss'], label='LR=0.001 Val Loss')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(history_01.history['accuracy'], label='LR=0.01 Train Acc')
plt.plot(history_01.history['val_accuracy'], label='LR=0.01 Val Acc')
plt.plot(history_001.history['accuracy'], label='LR=0.001 Train Acc')
plt.plot(history_001.history['val_accuracy'], label='LR=0.001 Val Acc')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
