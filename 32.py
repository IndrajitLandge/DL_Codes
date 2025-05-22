---Implement a CNN on Tomato dataset using batch sizes of 32 and 64 separately. Keep the learning rate 
fixed at 0.0001 and compare results. 

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data preparation: Assuming images are in 'train', 'validation', and 'test' directories
train_dir = "data/train"  # Path to training data
val_dir = "data/validation"  # Path to validation data

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Define CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # Adjust to match the number of classes
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train with batch size 32
train_generator_32 = datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)
val_generator_32 = datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)

model_32 = create_model()
history_32 = model_32.fit(
    train_generator_32,
    validation_data=val_generator_32,
    epochs=10
)

# Train with batch size 64
train_generator_64 = datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=64, class_mode='categorical'
)
val_generator_64 = datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=64, class_mode='categorical'
)

model_64 = create_model()
history_64 = model_64.fit(
    train_generator_64,
    validation_data=val_generator_64,
    epochs=10
)

# Plot training and validation curves
plt.figure(figsize=(12, 10))

# Training accuracy
plt.subplot(2, 2, 1)
plt.plot(history_32.history['accuracy'], label='Batch Size 32')
plt.plot(history_64.history['accuracy'], label='Batch Size 64')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Validation accuracy
plt.subplot(2, 2, 2)
plt.plot(history_32.history['val_accuracy'], label='Batch Size 32')
plt.plot(history_64.history['val_accuracy'], label='Batch Size 64')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Training loss
plt.subplot(2, 2, 3)
plt.plot(history_32.history['loss'], label='Batch Size 32')
plt.plot(history_64.history['loss'], label='Batch Size 64')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Validation loss
plt.subplot(2, 2, 4)
plt.plot(history_32.history['val_loss'], label='Batch Size 32')
plt.plot(history_64.history['val_loss'], label='Batch Size 64')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
