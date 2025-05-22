# Compare DNN training using Adam and SGD optimizers (both with a learning rate of 0.001) on the Wildfire dataset

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Paths
train_path = "wildfire_dataset/training"
val_test_path = "wildfire_dataset/test and val"

# Parameters
img_size = (150, 150)
batch_size = 32
epochs = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

test_generator = val_test_datagen.flow_from_directory(
    val_test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Model creation function
def create_model():
    model = Sequential([
        Flatten(input_shape=(150, 150, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Train Adam model
model_adam = create_model()
model_adam.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history_adam = model_adam.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=2
)

# Evaluate Adam on test set
test_loss_adam, test_acc_adam = model_adam.evaluate(test_generator)
print(f"\nAdam Test Accuracy: {test_acc_adam:.4f}")

# Predict & classification report Adam
y_pred_prob_adam = model_adam.predict(test_generator)
y_pred_adam = (y_pred_prob_adam > 0.5).astype(int).flatten()
y_true = test_generator.classes
print("\nAdam Classification Report:")
print(classification_report(y_true, y_pred_adam, target_names=list(test_generator.class_indices.keys())))

# Train SGD model
model_sgd = create_model()
model_sgd.compile(
    optimizer=SGD(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history_sgd = model_sgd.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=2
)

# Evaluate SGD on test set
test_loss_sgd, test_acc_sgd = model_sgd.evaluate(test_generator)
print(f"\nSGD Test Accuracy: {test_acc_sgd:.4f}")

# Predict & classification report SGD
y_pred_prob_sgd = model_sgd.predict(test_generator)
y_pred_sgd = (y_pred_prob_sgd > 0.5).astype(int).flatten()
print("\nSGD Classification Report:")
print(classification_report(y_true, y_pred_sgd, target_names=list(test_generator.class_indices.keys())))

# Plot accuracy comparison
plt.plot(history_adam.history['accuracy'], label='Adam - Train Acc')
plt.plot(history_adam.history['val_accuracy'], label='Adam - Val Acc')
plt.plot(history_sgd.history['accuracy'], label='SGD - Train Acc')
plt.plot(history_sgd.history['val_accuracy'], label='SGD - Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy: Adam vs SGD')
plt.legend()
plt.grid(True)
plt.show()
