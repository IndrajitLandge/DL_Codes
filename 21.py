---Statement 21  
Implement a CNN on Tomato dataset using batch sizes of 32 and 64 separately. Keep the learning 
rate fixed at 0.0001 and compare results.

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

data_dir = r"E:\DL_LAB_EXAM\Datasets\Plant_data\Tomato"
img_size = (128, 128)
learning_rate = 0.0001
epochs = 10

def load_data(batch_size):
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

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE))

def build_model(num_classes):
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
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_plot(batch_size):
    print(f"\nTraining with batch size: {batch_size}")
    train_ds, val_ds, test_ds = load_data(batch_size)
    model = build_model(num_classes=6)
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy (batch size {batch_size}): {test_acc*100:.2f}%")

    return history, test_acc

# Train with batch sizes 32 and 64
history_32, acc_32 = train_and_plot(32)
history_64, acc_64 = train_and_plot(64)

# Plotting accuracy comparison
plt.plot(history_32.history['val_accuracy'], label='Val Acc (Batch 32)')
plt.plot(history_64.history['val_accuracy'], label='Val Acc (Batch 64)')
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()
