# Use a batch size of 64 and learning rate of 0.001 to train a DNN on the UCI dataset. Document training accuracy and loss.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Load the dataset
columns = ["letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar",
           "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

df = pd.read_csv("E:/DL_LAB_EXAM/Datasets/UCI_letter_recognition/letter-recognition.data", names=columns)

# 2. Separate features and labels
X = df.iloc[:, 1:].values  # Features (skip 'letter')
y = df.iloc[:, 0].values   # Labels (letters A-Z)

# 3. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# 5. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 7. Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    verbose=2,
    validation_split=0.1
)

# 9. Final training performance
train_acc = history.history['accuracy'][-1]
train_loss = history.history['loss'][-1]
print(f"\nFinal Training Accuracy: {train_acc:.4f}")
print(f"Final Training Loss: {train_loss:.4f}")

# 10. Plot training and validation curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
