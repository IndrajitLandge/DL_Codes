# Preprocess the Alphabet CSV dataset using label encoding and standard scaling, then train a simple DNN using batch size 32 and learning rate 0.0001

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset (replace with your file path)
data = pd.read_csv('/path/to/alphabet_dataset.csv')

# Example: Assume last column is target (labels) and rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Label encode the target (for categorical string labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standard scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Number of classes for output layer
num_classes = len(label_encoder.classes_)

# Convert labels to categorical (one-hot)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Build a simple DNN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile with Adam optimizer and learning rate 0.0001
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model with batch size 32
model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20,
    batch_size=32,
    verbose=2
)
