---Statement 25 
Use LeNet architecture to classify the Cats and Dogs dataset, and plot training loss and accuracy 
curves.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 1. Dataset paths
train_dir = 'cats_and_dogs_filtered/train'
val_dir = 'cats_and_dogs_filtered/validation'

# 2. Load datasets
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 3. Normalize pixel values
normalizer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalizer(x), y))
val_ds = val_ds.map(lambda x, y: (normalizer(x), y))

# 4. LeNet Model
model = Sequential([
    Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(64, 64, 3)),
    AveragePooling2D(),
    Conv2D(16, (5, 5), activation='relu'),
    AveragePooling2D(),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

# 5. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 7. Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
