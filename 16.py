---Statement 16 
Multiclass classification using Deep Neural Networks: Example: Use the OCR letter recognition 
dataset/Alphabet.csv 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- Step 1: Load the dataset ---
data = pd.read_csv('Alphabet.csv')

# --- Step 2: Inspect data ---
print(data.head())
# Assume last column is 'label' (the letter), and rest are features

# --- Step 3: Prepare features and labels ---
X = data.drop('label', axis=1).values
y = data['label'].values

# --- Step 4: Encode labels to integers ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Step 5: One-hot encode labels for multiclass classification ---
y_cat = to_categorical(y_encoded)

# --- Step 6: Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 7: Train-test split ---
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# --- Step 8: Define DNN model ---
num_classes = y_cat.shape[1]
input_dim = X_train.shape[1]

model = Sequential([
    Dense(256, activation='relu', input_shape=(input_dim,)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Step 9: Train the model ---
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=30, batch_size=64, verbose=1)

# --- Step 10: Plot training history ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
