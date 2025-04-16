import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the pretrained model
model = load_model("efficientnet_deepfake_model.h5")

def preprocess_image(image_path):
    """Preprocess a single image for EfficientNet."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

def evaluate_folder(folder_path):
    """Evaluate model on images inside real/ and fake/ subfolders."""
    classes = ['real', 'fake']
    y_true = []
    y_pred = []

    for label in classes:
        class_folder = os.path.join(folder_path, label)
        if not os.path.exists(class_folder):
            print(f"⚠️ Folder not found: {class_folder}")
            continue

        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            img = preprocess_image(image_path)

            if img is None:
                print(f"❌ Could not load image: {image_path}")
                continue

            prediction = model.predict(img)[0][0]
            predicted_label = 'fake' if prediction > 0.5 else 'real'

            y_true.append(label)
            y_pred.append(predicted_label)

    return y_true, y_pred

def print_metrics(y_true, y_pred):
    """Print performance metrics."""
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['real', 'fake']))

    print("✅ Accuracy:", accuracy_score(y_true, y_pred))
    print("📌 Precision:", precision_score(y_true, y_pred, pos_label='fake'))
    print("📈 Recall:", recall_score(y_true, y_pred, pos_label='fake'))
    print("🏁 F1 Score:", f1_score(y_true, y_pred, pos_label='fake'))

    print("\n🧮 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=['real', 'fake']))

# Set your dataset folder path
dataset_path = "real_and_fake_face"
y_true, y_pred = evaluate_folder(dataset_path)
print_metrics(y_true, y_pred)
