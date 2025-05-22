---Build both CNN and DNN models for the CIFAR-10 dataset, compare their accuracy and loss 

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data (scale pixel values between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Build a simple DNN model
dnn_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train DNN
history_dnn = dnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=0)

# 4. Build a simple CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train CNN
history_cnn = cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=0)

# 6. Evaluate both models
dnn_loss, dnn_accuracy = dnn_model.evaluate(x_test, y_test, verbose=0)
cnn_loss, cnn_accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)

print(f"DNN Accuracy: {dnn_accuracy:.4f}, Loss: {dnn_loss:.4f}")
print(f"CNN Accuracy: {cnn_accuracy:.4f}, Loss: {cnn_loss:.4f}")

# 7. Plot Accuracy Comparison
plt.plot(history_dnn.history['val_accuracy'], label='DNN Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Accuracy')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 8. Plot Loss Comparison
plt.plot(history_dnn.history['val_loss'], label='DNN Loss')
plt.plot(history_cnn.history['val_loss'], label='CNN Loss')
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
