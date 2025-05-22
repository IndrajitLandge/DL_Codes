# Preprocess the Alphabet dataset and train a CNN with the architecture using Adam optimizer, 20 epochs, batch size 64, and learning rate 0.001.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load CSV dataset
data = pd.read_csv('/path/to/alphabet_dataset.csv')

# Assuming last column is label, rest are features (e.g., pixel values)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Label encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Number of classes
num_classes = len(np.unique(y_encoded))

# One-hot encode labels
y_encoded = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Scale features (pixel values usually from 0-255, so scale 0-1)
X = X / 255.0

# Assuming images are 28x28 pixels (modify if different)
img_height, img_width = 28, 28

# Reshape X to 4D tensor for CNN: (samples, height, width, channels)
X = X.reshape(-1, img_height, img_width, 1)  # 1 channel for grayscale

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile with Adam optimizer and learning rate 0.001
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with batch size 64 and 20 epochs
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    verbose=2
)
