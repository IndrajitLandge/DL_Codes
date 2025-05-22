---Statement 20 
Classify Apple leaf images using a CNN without data augmentation for 10 epochs. 

import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Dataset path
data_dir = r"E:/DL_LAB_EXAM/Datasets/Plant_data/Apple"

# Parameters
img_size = (128, 128)
batch_size = 32

# Load data without augmentation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "Train"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "Val"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "Test"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Simple CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")
