import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# SETTINGS

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
MAX_IMAGES_PER_CLASS = 200  # Adjust based on available RAM
DATASET_PATH = "/Users/harshitbansal/real_and_fake_face"
REAL_DIR = os.path.join(DATASET_PATH, "real")
FAKE_DIR = os.path.join(DATASET_PATH, "fake")

#  CLEANING + BALANCING
def is_valid_image(filepath):
    try:
        Image.open(filepath).verify()
        return True
    except:
        return False

def get_image_paths(directory, max_images):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    valid_images = [f for f in tqdm(files, desc=f"Cleaning {os.path.basename(directory)}") if is_valid_image(f)]
    return random.sample(valid_images, min(max_images, len(valid_images)))

real_images = get_image_paths(REAL_DIR, MAX_IMAGES_PER_CLASS)
fake_images = get_image_paths(FAKE_DIR, MAX_IMAGES_PER_CLASS)

all_images = real_images + fake_images
all_labels = [0]*len(real_images) + [1]*len(fake_images)  # 0 = real, 1 = fake

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# DATA LOADERS

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_paths, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# MODEL BUILDING
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# CALLBACKS
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("efficientnetB4_deepfake_best.keras", save_best_only=True)
]

#TRAIN

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# SAVE FINAL MODEL

model.save("efficientnetB4_deepfake_final.keras")
