import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------
TEST_REAL_DIR = 'test real and fake video/real'
TEST_FAKE_DIR = 'test real and fake video/fake'
MAX_FRAMES = 10
IMG_SIZE = 224
MODEL_PATH = 'cnn_gru_model.h5'
CSV_OUTPUT = 'detailed_predictions.csv'

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

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

    # Pad with empty frames if not enough
    if len(frames) < max_frames:
        frames += [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)] * (max_frames - len(frames))

    return np.array(frames)

# ---------------------------------------------
# PREDICT ONE VIDEO
# ---------------------------------------------
def predict_video(video_path):
    frames = extract_frames(video_path)
    input_tensor = np.expand_dims(frames, axis=0)
    prediction = model.predict(input_tensor, verbose=0)[0][0]
    return prediction

# ---------------------------------------------
# TEST AND LOG PREDICTIONS
# ---------------------------------------------
def evaluate_model():
    video_info = []
    y_true = []
    y_pred_probs = []

    # Load real videos
    for filename in os.listdir(TEST_REAL_DIR):
        if filename.endswith('.mp4'):
            video_info.append({
                "path": os.path.join(TEST_REAL_DIR, filename),
                "label": 0,
                "filename": filename
            })

    # Load fake videos
    for filename in os.listdir(TEST_FAKE_DIR):
        if filename.endswith('.mp4'):
            video_info.append({
                "path": os.path.join(TEST_FAKE_DIR, filename),
                "label": 1,
                "filename": filename
            })

    print(f"[INFO] Evaluating {len(video_info)} videos...\n")

    detailed_logs = []

    for video in tqdm(video_info):
        prob = predict_video(video["path"])
        pred_label = 1 if prob >= 0.5 else 0

        y_true.append(video["label"])
        y_pred_probs.append(prob)

        detailed_logs.append({
            "Filename": video["filename"],
            "True Label": "real" if video["label"] == 0 else "fake",
            "Predicted Probability": round(prob, 4),
            "Predicted Label": "real" if pred_label == 0 else "fake"
        })

        print(f"{video['filename']} | True: {detailed_logs[-1]['True Label']}, "
              f"Pred: {detailed_logs[-1]['Predicted Label']} ({prob:.4f})")

    # Compute metrics
    y_pred_labels = [1 if p >= 0.5 else 0 for p in y_pred_probs]
    acc = accuracy_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred_probs)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    print("\n[RESULTS]")
    print(f"Accuracy  : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Save detailed logs to CSV
    df = pd.DataFrame(detailed_logs)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\n[INFO] Detailed predictions saved to: {CSV_OUTPUT}")

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# ---------------------------------------------
# RUN
# ---------------------------------------------
if __name__ == "__main__":
    evaluate_model()
