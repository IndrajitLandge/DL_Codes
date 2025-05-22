---Statement 22 
Implement CNNs using Adam and RMSprop optimizers with a learning rate of 0.001 on Peach 
images. Record validation loss and accuracy. 

import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Path and settings
data_dir = r"E:\DL_LAB_EXAM\Datasets\Plant_data\Peach"
img_size = (128, 128)
batch_size = 32
epochs = 10
learning_rate = 0.001
num_classes = 2  # Bacterial Spot, Healthy

# Load data
def load_data():
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
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

# CNN model builder
def build_model(optimizer):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),
        layers.Conv2D(16, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and evaluate
def train_and_evaluate(optimizer_name):
    print(f"\nTraining with {optimizer_name} optimizer:")
    optimizer = tf.keras.optimizers.Adam(learning_rate) if optimizer_name == 'Adam' \
                else tf.keras.optimizers.RMSprop(learning_rate)
    
    train_ds, val_ds = load_data()
    model = build_model(optimizer)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    return val_loss, val_acc

# Run both experiments
adam_loss, adam_acc = train_and_evaluate("Adam")
rms_loss, rms_acc = train_and_evaluate("RMSprop")
