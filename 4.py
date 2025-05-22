# Use SGD optimizer with a learning rate of 0.01 to train a DNN on the Wildfire dataset, then evaluate precision, recall, and F1-score with supporting bar plots.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

# Paths to folders
train_path = 'wildfire_dataset/train'
test_path = 'wildfire_dataset/test'

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train and test images
train_data = train_datagen.flow_from_directory(train_path,
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

test_data = test_datagen.flow_from_directory(test_path,
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary',
                                             shuffle=False)

# Build DNN model
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile with SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, verbose=1)

# Predict on test data
y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Plot bar chart
metrics = ['Precision', 'Recall', 'F1-Score']
values = [precision, recall, f1]

plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.title('Evaluation Metrics')
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
