---Statement 18 
Implement a DNN using batch sizes 32 and 64 with a fixed learning rate of 0.001 on the UCI dataset. 
Compare model loss and performance. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time

# Load dataset
columns = ["letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar",
           "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
df = pd.read_csv("E:/DL_LAB_EXAM/Datasets/UCI_letter_recognition/letter-recognition.data", names=columns)

# Preprocess
X = df.drop(columns=['letter']).values
y = LabelEncoder().fit_transform(df['letter'])
X = StandardScaler().fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model builder function
def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Training function
def train_model(batch_size):
    model = build_model(X.shape[1], len(np.unique(y)))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    start = time.time()
    history = model.fit(X_train, y_train,
                        epochs=30,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=0)
    training_time = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    return history, training_time, test_acc, report

# Train models with batch size 32 and 64
hist_32, time_32, acc_32, report_32 = train_model(32)
hist_64, time_64, acc_64, report_64 = train_model(64)

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(hist_32.history['loss'], label='Train Loss (bs=32)')
plt.plot(hist_32.history['val_loss'], label='Val Loss (bs=32)')
plt.plot(hist_64.history['loss'], label='Train Loss (bs=64)', linestyle='--')
plt.plot(hist_64.history['val_loss'], label='Val Loss (bs=64)', linestyle='--')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_32.history['accuracy'], label='Train Acc (bs=32)')
plt.plot(hist_32.history['val_accuracy'], label='Val Acc (bs=32)')
plt.plot(hist_64.history['accuracy'], label='Train Acc (bs=64)', linestyle='--')
plt.plot(hist_64.history['val_accuracy'], label='Val Acc (bs=64)', linestyle='--')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Final comparison
print("=== Final Evaluation ===")
print(f"Batch Size 32: Accuracy = {acc_32:.4f}, Time = {time_32:.2f}s")
print(f"Batch Size 64: Accuracy = {acc_64:.4f}, Time = {time_64:.2f}s")
import seaborn as sns

# Data for bar plots
batch_sizes = ['32', '64']
accuracies = [acc_32, acc_64]
times = [time_32, time_64]

# Accuracy Bar Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=batch_sizes, y=accuracies, palette='viridis')
plt.title('Test Accuracy by Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # accuracy is between 0 and 1
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')

# Time Bar Plot
plt.subplot(1, 2, 2)
sns.barplot(x=batch_sizes, y=times, palette='magma')
plt.title('Training Time by Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Time (seconds)')
for i, t in enumerate(times):
    plt.text(i, t + 1, f"{t:.2f}s", ha='center')

plt.tight_layout()
plt.show()
