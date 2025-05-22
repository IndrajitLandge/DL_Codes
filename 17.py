---Statement 17 
Implement the training of a DNN using Adam and SGD optimizers with a learning rate of 0.001 on 
the Wildfire dataset. Provide comparative plots. 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD

# Load image data
train_dir = './Datasets/forest_fire/Training and Validation'
test_dir = './Datasets/forest_fire/Testing'

# Preprocess images
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
)
# Simple model function
def create_model():
    model = Sequential([
        Flatten(input_shape=(64, 64, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
model_adam = create_model()
model_adam.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_adam = model_adam.fit(train_generator, epochs=15, validation_data=test_generator, verbose=1)

# Train with SGD
model_sgd = create_model()
model_sgd.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_sgd = model_sgd.fit(train_generator, epochs=15, validation_data=test_generator, verbose=1)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history_adam.history['accuracy'], label='Adam - Train Acc')
plt.plot(history_adam.history['val_accuracy'], label='Adam - Val Acc')
plt.plot(history_sgd.history['accuracy'], label='SGD - Train Acc')
plt.plot(history_sgd.history['val_accuracy'], label='SGD - Val Acc')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history_adam.history['loss'], label='Adam - Train Loss')
plt.plot(history_adam.history['val_loss'], label='Adam - Val Loss')
plt.plot(history_sgd.history['loss'], label='SGD - Train Loss')
plt.plot(history_sgd.history['val_loss'], label='SGD - Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
