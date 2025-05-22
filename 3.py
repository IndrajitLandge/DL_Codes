# Train a Deep Neural Network on the MNIST dataset using RMSprop optimizer with a learning rate of 0.0001, and compare results using an accuracy table and ROC curve.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
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

# Compile the model with RMSprop optimizer
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_cat, epochs=5, batch_size=32, verbose=1,
                    validation_data=(x_test, y_test_cat))

# Predict
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Accuracy Table
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print("\nAccuracy Table:")
print(f"Training Accuracy:  {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# ROC AUC score (multi-class)
roc_auc = roc_auc_score(y_test_cat, y_pred_probs, multi_class='ovr')
print(f"Multi-class ROC AUC Score: {roc_auc:.4f}")

# ROC curve for class 0
fpr, tpr, _ = roc_curve(y_test_cat[:, 0], y_pred_probs[:, 0])
plt.plot(fpr, tpr, label='Class 0')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Class 0')
plt.legend()
plt.grid(True)
plt.show()
