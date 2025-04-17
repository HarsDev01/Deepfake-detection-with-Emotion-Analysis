import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from deepface import DeepFace
from collections import Counter

# Load the EfficientNetB4-based deepfake model
model = load_model("efficientnetB4_deepfake_final.keras")

# Emotion categories
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "unknown"]
emotion_to_index = {e: i for i, e in enumerate(EMOTION_LABELS)}

# Configuration
FRAME_SKIP = 5
SMOOTH_WINDOW = 5
AVG_THRESHOLD = 0.45
MAX_THRESHOLD = 0.6
HIGH_CONFIDENCE_COUNT = 10
PREDICTION_THRESHOLD = 0.5  # Used for per-frame judgment

# Face extraction using Haar Cascade
def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Preprocess the face for EfficientNet input
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = preprocess_input(face_img.astype(np.float32))
    return np.expand_dims(face_img, axis=0)

# Predict deepfake probability for a face
def predict_frame_face(face_img):
    preprocessed = preprocess_face(face_img)
    return float(model.predict(preprocessed, verbose=0)[0][0])

# Detect emotion using DeepFace
def get_emotion(face_img):
    try:
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion'].lower()
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
        return "unknown"

# Main video analysis function
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    frame_count = 0
    predictions = []
    frame_indices = []
    emotions = []

    print("🔍 Starting analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        faces = extract_faces(frame)
        if len(faces) == 0:
            continue

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            prob = predict_frame_face(face_img)
            emotion = get_emotion(face_img)

            predictions.append(prob)
            emotions.append(emotion)
            frame_indices.append(frame_count)

            print(f"🧪 Frame {frame_count} | Deepfake Probability: {prob:.4f} | Emotion: {emotion}")

    cap.release()

    if not predictions:
        print("⚠️ No faces detected.")
        return

    # Compute metrics
    avg_prob = np.mean(predictions)
    max_prob = np.max(predictions)
    num_above_thresh = sum(p > PREDICTION_THRESHOLD for p in predictions)

    print(f"\n🧠 Average Deepfake Probability: {avg_prob:.4f}")
    print(f"🔥 Max Deepfake Probability: {max_prob:.4f}")
    print(f"📊 Frames > {PREDICTION_THRESHOLD}: {num_above_thresh}")

    # Enhanced Decision Logic
    if avg_prob > AVG_THRESHOLD or max_prob > MAX_THRESHOLD or num_above_thresh >= HIGH_CONFIDENCE_COUNT:
        print("🚨 The video is likely a deepfake.")
    else:
        print("✅ The video is likely real.")

    # Smoothed predictions
    smoothed_preds = pd.Series(predictions).rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

    # Plot Deepfake Probabilities
    plt.figure(figsize=(10, 4))
    plt.plot(frame_indices, predictions, linestyle='--', color='lightcoral', label="Raw Prob")
    plt.plot(frame_indices, smoothed_preds, color='red', label=f"Smoothed (w={SMOOTH_WINDOW})")
    plt.axhline(PREDICTION_THRESHOLD, color='green', linestyle='--', label=f"Threshold ({PREDICTION_THRESHOLD})")
    plt.title("Deepfake Detection Probability Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("deepfake_video_graph.png")
    plt.show()

    # Plot Emotion Timeline
    emotion_codes = [emotion_to_index.get(e, emotion_to_index["unknown"]) for e in emotions]
    plt.figure(figsize=(10, 4))
    plt.plot(frame_indices, emotion_codes, marker='o', linestyle='-', color='blue')
    plt.yticks(list(emotion_to_index.values()), list(emotion_to_index.keys()))
    plt.title("Emotion Detection Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Emotion")
    plt.tight_layout()
    plt.savefig("emotion_over_time_graph.png")
    plt.show()

    # Print emotion stats
    print("\n🎭 Emotion Distribution:")
    for emotion, count in Counter(emotions).most_common():
        print(f"   - {emotion}: {count} times")

# Set path to your video
video_path = "test_video.mp4"
analyze_video(video_path)
