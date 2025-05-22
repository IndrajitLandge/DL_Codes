#Implement a DNN using RMSprop with learning rates 0.01 and 0.0001 on the Wildfire dataset. Compare training and validation performance. 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

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
def create_model(lr):
    model = Sequential([
        Flatten(input_shape=(150, 150, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train with lr = 0.01
model_high_lr = create_model(0.01)
history_high = model_high_lr.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=2
)

# Train with lr = 0.0001
model_low_lr = create_model(0.0001)
history_low = model_low_lr.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=2
)

# Plot results
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_high.history['accuracy'], label='LR=0.01 - Train')
plt.plot(history_high.history['val_accuracy'], label='LR=0.01 - Val')
plt.plot(history_low.history['accuracy'], label='LR=0.0001 - Train')
plt.plot(history_low.history['val_accuracy'], label='LR=0.0001 - Val')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_high.history['loss'], label='LR=0.01 - Train')
plt.plot(history_high.history['val_loss'], label='LR=0.01 - Val')
plt.plot(history_low.history['loss'], label='LR=0.0001 - Train')
plt.plot(history_low.history['val_loss'], label='LR=0.0001 - Val')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
