import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from deepface import DeepFace

# Load the fine-tuned EfficientNetB4 model
model = load_model("efficientnetB4_deepfake_final.keras")

# Define emotion detection using DeepFace
def get_emotion(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
        return result[0]["dominant_emotion"].lower()
    except Exception:
        return "unknown"

# Deepfake + Emotion Prediction on Single Image
def predict_deepfake_and_emotion(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Image not found!")
        return

    # Convert to RGB for DeepFace and Keras
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # DeepFace emotion detection (original size)
    emotion = get_emotion(image_rgb)

    # Resize and preprocess for EfficientNetB4
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_preprocessed = preprocess_input(image_resized.astype(np.float32))
    image_batch = np.expand_dims(image_preprocessed, axis=0)

    # Run deepfake prediction
    prediction = model.predict(image_batch)[0][0]

    print(f"🔍 Deepfake Probability: {prediction:.4f}")
    print(f"🎭 Detected Emotion: {emotion}")

    if prediction > 0.5:
        print("🚨 This image is likely a deepfake!")
    else:
        print("✅ This image is likely real.")

# Test it on your image
image_path = "test_fake_image.jpg"  # Update with your image path
predict_deepfake_and_emotion(image_path)
