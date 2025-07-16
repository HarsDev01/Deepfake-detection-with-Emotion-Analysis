# 🧠 Deepfake Detection with Emotion Analysis

> A hybrid deep learning system to detect deepfakes in images and videos using EfficientNetB4 and CNN-GRU, combined with facial emotion analysis using DeepFace. Designed to be fast, scalable, and suitable for real-world deployment.

---

## 🚀 Features

* ✅ Detects deepfakes in both **images** and **videos**
* 📊 Generates **real-time metrics**: Accuracy, Precision, Recall, F1, AUC
* 🎭 Performs **emotion recognition** on detected faces using DeepFace
* ⚙️ Uses **EfficientNetB4** for feature extraction
* 🧠 Includes **CNN + GRU model** for temporal video analysis
* 📈 Live plots for deepfake probability and emotion timeline
* 💻 Lightweight, efficient & optimized for low-resource environments

---

## 🧰 Tech Stack

| Area             | Tools & Libraries                      |
| ---------------- | -------------------------------------- |
| Deep Learning    | TensorFlow, Keras, EfficientNetB4, GRU |
| Face Analysis    | DeepFace (for emotion detection)       |
| Image Processing | OpenCV, Haar Cascades                  |
| Data             | NumPy, pandas, scikit-learn            |
| Visualization    | Matplotlib                             |

---

## 📁 Project Structure

```bash
.
├── models/
│   ├── efficientnetB4_deepfake_final.keras
│   ├── cnn_gru_model.h5
│
├── real_and_fake_face/
│   ├── real/
│   └── fake/
│
├── real and fake video/
│   ├── real/
│   └── fake/
│
├── test real and fake video/
│   ├── real/
│   └── fake/
│
├── deepfake_train.py                   # Train EfficientNetB4 image model
├── evaluate_image_model.py            # Evaluate on static images
├── analyze_single_image.py            # Image deepfake + emotion detection
├── analyze_video_with_emotion.py      # Video deepfake + emotion + graph
├── cnn_gru_train.py                   # Train CNN + GRU on video frames
├── cnn_gru_evaluation.py              # Evaluate videos & log predictions
```

---

## 🧠 Model Architectures

### 1. **EfficientNetB4 (Image-Based)**

```text
EfficientNetB4 (frozen)
→ GlobalAveragePooling2D
→ Dropout + Dense
→ Sigmoid output (real vs fake)
```

### 2. **CNN + GRU (Video-Based)**

```text
TimeDistributed(EfficientNetB4)
→ GlobalAveragePooling2D (per frame)
→ GRU layer
→ Dropout + Dense
→ Sigmoid output
```

---

## 🧪 Evaluation Metrics

| Model                 | Accuracy  | Precision | Recall    | F1 Score  |
| --------------------- | --------- | --------- | --------- | --------- |
| EfficientNetB4 (Base) | 48.1%     | 45.6%     | 53.5%     | 49.2%     |
| **Fine-tuned Model**  | **57.7%** | **53.6%** | **74.6%** | **62.3%** |


---

## 🎯 Emotion Analysis

* Performed using **DeepFace**
* Extracted directly from **video frames** or static images
* Tracked over time in videos with frame-wise stats and plots

🎭 Emotions supported: `happy`, `sad`, `angry`, `surprise`, `neutral`, etc.

---

## 📈 Output Graphs

* **Deepfake Probability Timeline** across video frames
* **Emotion Timeline Plot** to show mood shifts over time
* Saved as:

  * `deepfake_video_graph.png`
  * `emotion_over_time_graph.png`

---

## 🛠️ How to Run

### ▶️ Train Image-Based Model (EfficientNetB4)

```bash
python deepfake_train.py
```

### 🧪 Evaluate Image-Based Model

```bash
python evaluate_image_model.py
```

### 📸 Analyze Deepfake + Emotion (Single Image)

```bash
python analyze_single_image.py
```

### 🎥 Analyze Deepfake + Emotion (Video)

```bash
python analyze_video_with_emotion.py
```

### 🎞️ Train Temporal Model (CNN-GRU on Video)

```bash
python cnn_gru_train.py
```

### 🧪 Evaluate Temporal Model on Video Dataset

```bash
python cnn_gru_evaluation.py
```

---

## 📂 Dataset Used

* `real_and_fake_face/` — manually balanced dataset with real & fake face images
* `real and fake video/` — training videos categorized as real/fake
* `test real and fake video/` — separate test set for generalization

Data preprocessing includes:

* Image validation and cleaning
* Frame extraction with resizing
* Haar cascade face detection

---

## 📌 Results Summary

* CNN-GRU video model improves temporal understanding vs. static image detection.
* Emotion analysis adds context to deepfake results.
* Decision logic includes:

  * Average probability
  * Max probability
  * Smoothed curve thresholding

---

## 📊 Sample Output

```bash
Frame 120 | Deepfake Probability: 0.8723 | Emotion: neutral  
Frame 125 | Deepfake Probability: 0.9231 | Emotion: happy  

Average Deepfake Probability: 0.89  
Likely: 🚨 Deepfake  
```

---

## 📌 Future Scope

* Add **race, age, and gender** analysis using DeepFace
* Deploy model using **Streamlit** or **Flask Web App**
* Use **Hugging Face Hub** for model sharing
* Add **real-time webcam streaming analysis**

---

## 👨‍💻 Author

**Harshit Bansal**
📫 [harshitbansalmi@gmail.com](mailto:harshitbansalmi@gmail.com)
🔗 [GitHub](https://github.com/HarsDev01) | [LinkedIn](https://linkedin.com/in/harshit-bansal-928916369)

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).  
Feel free to use, adapt, and cite the work with attribution.

---

