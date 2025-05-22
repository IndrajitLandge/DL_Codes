""" Evaluating DNN on CIFAR-10 Using Batch Size Variation
● Load CIFAR-10 dataset
● Use a feed-forward network with BatchNormalization
● Train with batch sizes 32 and 64, keeping other parameters constant
● Use Adam optimizer and train for 10 epochs
● Compare accuracy and plot graphs """

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten y labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Function to create model with BatchNorm
def create_model():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

# Function to train model with given batch size
def train_model(batch_size):
    model = create_model()
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=batch_size,
        verbose=0
    )
    return history

# Train with batch size 32
history_32 = train_model(32)

# Train with batch size 64
history_64 = train_model(64)

# Plot accuracy comparison
plt.plot(history_32.history['accuracy'], label='Batch Size 32 Train Acc')
plt.plot(history_32.history['val_accuracy'], label='Batch Size 32 Val Acc')
plt.plot(history_64.history['accuracy'], label='Batch Size 64 Train Acc')
plt.plot(history_64.history['val_accuracy'], label='Batch Size 64 Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.show()
