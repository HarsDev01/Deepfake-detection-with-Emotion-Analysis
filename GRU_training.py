import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, GRU, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.metrics import AUC
import pandas as pd

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------
VIDEO_DIR_REAL = 'real and fake video/real'
VIDEO_DIR_FAKE = 'real and fake video/fake'
MAX_FRAMES = 10
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10

# ---------------------------------------------
# FRAME EXTRACTION FUNCTION
# ---------------------------------------------
def extract_frames(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

# ---------------------------------------------
# LOAD DATASET
# ---------------------------------------------
def load_dataset(real_dir, fake_dir):
    X = []
    y = []

    real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')]

    print("[INFO] Loading real videos...")
    for video_path in tqdm(real_videos):
        frames = extract_frames(video_path)
        if len(frames) == MAX_FRAMES:
            X.append(frames)
            y.append(0)

    print("[INFO] Loading fake videos...")
    for video_path in tqdm(fake_videos):
        frames = extract_frames(video_path)
        if len(frames) == MAX_FRAMES:
            X.append(frames)
            y.append(1)

    return np.array(X), np.array(y)

# ---------------------------------------------
# BUILD CNN + GRU MODEL
# ---------------------------------------------
def build_model():
    cnn_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    cnn_base.trainable = False  # Freeze CNN layers

    frame_input = Input(shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    x = TimeDistributed(cnn_base)(frame_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = GRU(128)(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=frame_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

    return model

# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------
if __name__ == "__main__":
    # Step 1: Load data
    X, y = load_dataset(VIDEO_DIR_REAL, VIDEO_DIR_FAKE)
    print(f"[INFO] Loaded dataset shape: X={X.shape}, y={y.shape}")

    # Step 2: Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 3: Build model
    model = build_model()
    model.summary()

    # Step 4: Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("cnn_gru_best_model.h5", save_best_only=True)
        ]
    )

    # Step 5: Evaluate model
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val)
    print(f"[RESULT] Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}")

    # Step 6: Save model and results
    model.save('cnn_gru_model.h5')
    results = pd.DataFrame({
        'Model': ['EfficientNetB4 + GRU'],
        'Accuracy': [val_acc],
        'AUC': [val_auc]
    })
    results.to_csv('model_comparison.csv', index=False)

    print("[INFO] Model and results saved locally.")
