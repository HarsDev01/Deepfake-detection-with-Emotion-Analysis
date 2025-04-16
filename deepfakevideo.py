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

# Map emotions to numeric codes for plotting
EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "unknown"
]
emotion_to_index = {e: i for i, e in enumerate(EMOTION_LABELS)}

def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return faces

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = preprocess_input(face_img.astype(np.float32))
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def predict_frame_face(face_img):
    preprocessed = preprocess_face(face_img)
    prediction = model.predict(preprocessed, verbose=0)[0][0]
    return prediction

def get_emotion(face_img):
    try:
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion'].lower()
    except Exception:
        return "unknown"

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    frame_count = 0
    predictions = []
    frame_indices = []
    emotions = []

    print("🔍 Analyzing video frame-by-frame...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
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

    if len(predictions) == 0:
        print("⚠️ No faces detected in video.")
        return

    avg_prob = np.mean(predictions)
    print(f"\n🧠 Average Deepfake Probability: {avg_prob:.4f}")

    if avg_prob > 0.5:
        print("🚨 The video is likely a deepfake.")
    else:
        print("✅ The video is likely real.")

    # Deepfake Probability Plot (Smoothed)
    window_size = 5
    smoothed_preds = pd.Series(predictions).rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(frame_indices, predictions, color='lightcoral', linestyle='--', label='Raw Deepfake Probabilities')
    plt.plot(frame_indices, smoothed_preds, color='red', label=f'Smoothed (Window={window_size})')
    plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Frame Number')
    plt.ylabel('Deepfake Probability')
    plt.title('Deepfake Detection Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig("deepfake_video_graph.png")
    plt.show()

    # Emotion Over Time Plot
    emotion_codes = [emotion_to_index.get(e, emotion_to_index["unknown"]) for e in emotions]
    plt.figure(figsize=(10, 4))
    plt.plot(frame_indices, emotion_codes, marker='o', linestyle='-', color='blue')
    plt.yticks(list(emotion_to_index.values()), list(emotion_to_index.keys()))
    plt.xlabel('Frame Number')
    plt.ylabel('Detected Emotion')
    plt.title('Emotion Detection Over Time')
    plt.tight_layout()
    plt.savefig("emotion_over_time_graph.png")
    plt.show()

    # Emotion Distribution Summary
    print("\n🎭 Emotion Distribution:")
    emotion_counts = Counter(emotions)
    for emotion, count in emotion_counts.most_common():
        print(f"   - {emotion}: {count} times")


# Set your test video path
video_path = "test_video.mp4"
analyze_video(video_path)
