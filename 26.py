---Statement 26  
Use MobileNet architecture  perform transfer learning on the Cats and Dogs dataset, and evaluate 
model performance using a classification report. 

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# 1. Set parameters
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# 2. ImageDataGenerator with normalization
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

# 3. Load images
train_dir = 'cats_and_dogs_filtered/train'
val_dir = 'cats_and_dogs_filtered/validation'

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# 4. Load base model
base_model = MobileNetV2(input_shape=(160, 160, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

# 5. Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# 6. Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 7. Train the model
history = model.fit(train_data, validation_data=val_data, epochs=5)

# 8. Classification report
y_true = val_data.classes
y_pred = model.predict(val_data)
y_pred = np.round(y_pred).astype(int).flatten()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))
