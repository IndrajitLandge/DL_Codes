# Compare the performance of a CNN and a DNN on the CIFAR-10 dataset. Highlight differences in accuracy and training time

import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encode labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# DNN model (Flatten input and fully connected layers)
def build_dnn():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# CNN model (Conv + Pooling + Dense layers)
def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and time DNN
dnn = build_dnn()
start = time.time()
dnn_history = dnn.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=2)
dnn_time = time.time() - start
dnn_loss, dnn_acc = dnn.evaluate(X_test, y_test, verbose=0)

# Train and time CNN
cnn = build_cnn()
start = time.time()
cnn_history = cnn.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=2)
cnn_time = time.time() - start
cnn_loss, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)

# Results
print(f"DNN Test Accuracy: {dnn_acc:.4f}, Training Time: {dnn_time:.2f} sec")
print(f"CNN Test Accuracy: {cnn_acc:.4f}, Training Time: {cnn_time:.2f} sec")
