---Statement 31 
Implement LSTM models on GOOGL.csv with learning rates 0.001 and 0.0001 for 20 and 50 epochs. 
Compare accuracy and convergence.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the GOOGL.csv dataset
data = pd.read_csv("GOOGL.csv")
print(data.head())  # Inspect the data

# Use 'Close' price for prediction
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Create sequences for LSTM
def create_sequences(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 50
X, y = create_sequences(prices_scaled, time_steps)

# Split into training and testing datasets
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Function to build LSTM model
def build_lstm_model(learning_rate):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Train and evaluate models
learning_rates = [0.001, 0.0001]
epochs_list = [20, 50]

results = {}

for lr in learning_rates:
    for epochs in epochs_list:
        model = build_lstm_model(learning_rate=lr)
        print(f"\nTraining model with LR={lr}, Epochs={epochs}")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        results[(lr, epochs)] = history.history

# Plot accuracy and convergence
plt.figure(figsize=(14, 10))

for idx, (lr, epochs) in enumerate(results.keys()):
    history = results[(lr, epochs)]
    plt.subplot(2, 2, idx + 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"LR={lr}, Epochs={epochs}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()
