---Use transfer learning with VGG16 on the Cats and Dogs dataset, freezing the first 4 layers, and train 
the classifier and evaluate model performance using a classification report.

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# Set image size and paths
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = 'cats_and_dogs_filtered/train'
val_dir = 'cats_and_dogs_filtered/validation'

# Image generators with normalization
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir,
                                           target_size=IMG_SIZE,
                                           batch_size=BATCH_SIZE,
                                           class_mode='binary',
                                           shuffle=True)

val_data = val_gen.flow_from_directory(val_dir,
                                       target_size=IMG_SIZE,
                                       batch_size=BATCH_SIZE,
                                       class_mode='binary',
                                       shuffle=False)

# Load VGG16 base model without top
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze only the first 4 layers
for layer in base_model.layers[:4]:
    layer.trainable = False
for layer in base_model.layers[4:]:
    layer.trainable = True

# Add custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# Classification report
y_true = val_data.classes
y_pred = model.predict(val_data)
y_pred = np.round(y_pred).astype(int).flatten()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Cat", "Dog"]))
