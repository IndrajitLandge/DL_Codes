---Statement 19 
Preprocess the Alphabet dataset and train both a DNN and a CNN. Use Adam optimizer with a batch 
size of 64. Compare accuracy across 20 epochs. 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# --- Step 1: Load and preprocess the dataset ---
# For demonstration, let's create dummy data
# Replace this with your actual Alphabet dataset loading
num_classes = 26
img_height, img_width, channels = 28, 28, 1  # assuming grayscale images 28x28

# Dummy data: 10000 samples
X = np.random.rand(10000, img_height, img_width, channels).astype('float32')
y = np.random.randint(0, num_classes, 10000)

# Normalize pixel values
X /= 255.0

# One-hot encode labels
y = to_categorical(y, num_classes)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 2: Define the DNN model (flatten + dense layers) ---
def create_dnn():
    model = Sequential([
        Flatten(input_shape=(img_height, img_width, channels)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- Step 3: Define the CNN model ---
def create_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- Step 4: Compile and train models ---
batch_size = 64
epochs = 20

# Train DNN
dnn_model = create_dnn()
dnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training DNN...")
history_dnn = dnn_model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=batch_size,
                            epochs=epochs)

# Train CNN
cnn_model = create_cnn()
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining CNN...")
history_cnn = cnn_model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=batch_size,
                            epochs=epochs)

# --- Step 5: Compare accuracy ---
import matplotlib.pyplot as plt

plt.plot(history_dnn.history['val_accuracy'], label='DNN Validation Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
