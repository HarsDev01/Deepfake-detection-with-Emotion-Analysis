import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm

# Load your trained .keras model
model = load_model("efficientnetB4_deepfake_final.keras")

# Paths to real and fake folders
real_dir = "real_and_fake_face/real"
fake_dir = "real_and_fake_face/fake"

# Initialize lists for predictions and labels
y_true = []
y_pred = []

# Function to preprocess image
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Process real images
print("🔍 Evaluating real images...")
for filename in tqdm(os.listdir(real_dir)):
    img_path = os.path.join(real_dir, filename)
    img = preprocess_image(img_path)
    if img is None:
        continue
    prediction = model.predict(img)[0][0]
    y_pred.append(0 if prediction < 0.5 else 1)
    y_true.append(0)

# Process fake images
print("🔍 Evaluating fake images...")
for filename in tqdm(os.listdir(fake_dir)):
    img_path = os.path.join(fake_dir, filename)
    img = preprocess_image(img_path)
    if img is None:
        continue
    prediction = model.predict(img)[0][0]
    y_pred.append(0 if prediction < 0.5 else 1)
    y_true.append(1)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

# Report
print("\n✅ Evaluation Results:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\n🧾 Confusion Matrix:\n", cm)
print("\n📋 Classification Report:\n", classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
