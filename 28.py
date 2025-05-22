---statement 28 
Implement an RNN on the GOOGL.csv dataset and compare its training time and loss curve with an 
LSTM model. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# 1. Load data
df = pd.read_csv("GOOGL.csv")
data = df['Close'].values.reshape(-1, 1)

# 2. Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Prepare training sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(scaled_data, seq_length)

# 4. Split into training data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]

# Reshape input for RNN/LSTM: [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# 5. Train RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(seq_length, 1)))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mse')

start_rnn = time.time()
history_rnn = rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
end_rnn = time.time()

# 6. Train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(seq_length, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

start_lstm = time.time()
history_lstm = lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
end_lstm = time.time()

# 7. Compare training time
print(f"RNN Training Time:  {end_rnn - start_rnn:.2f} seconds")
print(f"LSTM Training Time: {end_lstm - start_lstm:.2f} seconds")

# 8. Plot loss curves
plt.plot(history_rnn.history['loss'], label='RNN Loss')
plt.plot(history_lstm.history['loss'], label='LSTM Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()
