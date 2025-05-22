#Train a DNN using the SGD optimizer with a learning rate of 0.0001 on the MNIST dataset and analyze the model's performance.

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile with SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_cat, epochs=5, batch_size=32, verbose=1)

# Predict
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print("Test Accuracy:", accuracy)
