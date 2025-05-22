---Statement 24 
Split Grape image data into 70% train, 15% validation, and 15% test. Train a CNN for 10 epochs 
using a fixed learning rate of 0.001.

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling

# 1. Load and split grape dataset
dataset_path = "grape_dataset"  # Update this to your folder path

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32)

val_batches = int(len(val_test_ds) * 0.5)
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

# 2. Normalize pixel values
train_ds = train_ds.map(lambda x, y: (Rescaling(1./255)(x), y))
val_ds = val_ds.map(lambda x, y: (Rescaling(1./255)(x), y))
test_ds = test_ds.map(lambda x, y: (Rescaling(1./255)(x), y))

# 3. Build simple CNN model
model = tf.keras.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_ds.class_names), activation='softmax')
])

# 4. Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6. Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.2f}")

# 7. Plot accuracy curve
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
