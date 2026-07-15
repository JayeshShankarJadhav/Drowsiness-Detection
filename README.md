# 😴 Drowsiness Detection System

A real-time, AI-powered driver safety system that monitors eye state via webcam and triggers an audio alarm the moment signs of drowsiness are detected — built using **OpenCV**, **Haar Cascade classifiers**, and a **Convolutional Neural Network (CNN)**.

Built as a semester project (Deep Learning) for MSc (Data Analytics), Pillai College of Arts, Commerce & Science (Autonomous), New Panvel — University of Mumbai, 2024–25.

---

## 📖 Overview

Drowsy driving is a major cause of road accidents — as dangerous as drunk driving, since it slows reaction time and impairs judgment. Long-distance drivers, night-shift workers, and people with sleep disorders are especially at risk, and manual/self-monitoring is unreliable since fatigue can set in suddenly and go unnoticed.

This system continuously monitors a driver's face and eyes through a webcam, classifies eye state in real time using a trained CNN, tracks a rolling "drowsiness score," and sounds an alarm once that score crosses a threshold — giving the driver a chance to react before a fatigue-related accident occurs.

---

## ✨ Features

- Real-time face & eye detection using Haar Cascade classifiers
- CNN-based eye-state classification (Open / Closed, and Yawn detection)
- Rolling drowsiness score with threshold-based alerting
- Audio alarm (via Pygame) triggered on sustained eye closure
- Visual on-screen feedback — flashing red border + live score overlay
- Lightweight enough for real-time webcam inference on a standard laptop

---

## 🧱 Tech Stack

- **Python** — core language for training and real-time processing
- **OpenCV** — face/eye detection, frame preprocessing, real-time video capture
- **TensorFlow / Keras** — CNN model architecture, training, and inference
- **Haar Cascade Classifiers** — pretrained face, left-eye, and right-eye detectors
- **Pygame** — plays the alarm sound on drowsiness detection
- **NumPy** — array/image manipulation
- **scikit-learn, Seaborn, Matplotlib** — evaluation (classification report, confusion matrix)

---

## 🧠 How It Works

1. Capture real-time video from the webcam.
2. Detect the face and both eyes per frame using Haar Cascade classifiers.
3. Crop and preprocess each eye region (grayscale, resized to 48×48, normalized).
4. Feed each eye image into the trained CNN, which classifies it as **Open** or **Closed**.
5. If both eyes are predicted closed, the drowsiness score increases each frame; if open, it decreases (floored at 0).
6. Once the score exceeds a threshold (15 consecutive "closed" frames), the system:
   - Plays an alarm sound
   - Flashes a growing/shrinking red border around the video feed
   - Saves a snapshot of the drowsy frame (`drowsy_image.jpg`)

---

## 📂 Dataset & Trained Model

Due to GitHub's file size limits, the training dataset and trained model weights (`drowsiness_model.keras`) are hosted externally rather than in this repository.

📁 **Dataset & Model Weights (Google Drive):** [Add your Google Drive link here](https://drive.google.com/your-link-here)

> Replace the link above with your actual shareable Google Drive link (Anyone with the link → Viewer).

The model was trained on a labeled 4-class dataset — **Closed**, **Open**, **no_yawn**, **yawn** — of grayscale 48×48 eye/face crops.

---

## 📁 Project Structure

```
Drowsiness-Detection-System/
├── drowsiness_detection.py       # Real-time webcam detection + alarm
├── model.py                       # Model evaluation (accuracy, classification report, confusion matrix)
├── testmodel.py                   # Single-image prediction test script
├── models/
│   └── drowsiness_model.keras     # Trained CNN (download from Drive link above)
├── haar cascade files/
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
├── alarm.wav                      # Alert sound
├── data/
│   └── test/                      # Test split (download from Drive link above)
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install opencv-python tensorflow numpy pygame scikit-learn seaborn matplotlib

# 4. Download the trained model from the Google Drive link above
#    and place drowsiness_model.keras inside models/

# 5. Run real-time detection
python drowsiness_detection.py
```

Press **`q`** to exit the live detection window.

---

## 🚀 Usage

- **Live detection:** run `drowsiness_detection.py` — a webcam window opens showing live eye-state classification, a running drowsiness score, and a flashing red alert border with alarm sound once drowsiness is confirmed.
- **Single-image test:** run `testmodel.py` with a path to any face image to see the model's predicted class (Open / Closed / no_yawn / yawn) overlaid on the image.
- **Model evaluation:** run `model.py` to reproduce the accuracy metrics and confusion matrix against the test set.

---

## 📊 Model Performance

Evaluated on a held-out test set of 433 images across 4 classes:

| Metric | Score |
|---|---|
| **Test Accuracy** | 78.52% |
| **Test Loss** | 0.4686 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Closed | 0.97 | 1.00 | 0.99 | 109 |
| Open | 1.00 | 0.96 | 0.98 | 109 |
| no_yawn | 0.67 | 0.37 | 0.47 | 109 |
| yawn | 0.55 | 0.81 | 0.66 | 106 |
| **Accuracy** | | | **0.79** | 433 |
| Macro avg | 0.80 | 0.79 | 0.77 | 433 |
| Weighted avg | 0.80 | 0.79 | 0.78 | 433 |

The model is highly reliable at the core task — distinguishing **open vs. closed eyes** (0.97–1.00 precision/recall) — which is what actually drives the real-time alarm logic. Yawn/no-yawn classification is noticeably weaker and is the clearest area for future improvement.

---

## 🔭 Future Enhancements

- Night-time / low-light detection using infrared cameras or low-light image enhancement
- Integration with vehicle IoT systems (steering vibration, automatic braking triggers)
- Transfer learning with MobileNet/EfficientNet to improve yawn-detection accuracy and inference speed
- Real-time cloud-based fatigue reporting and driver behavior analytics
- Emotion/stress recognition to extend beyond pure drowsiness detection

---

## 🙏 Acknowledgements

Developed by **Jayesh Jadhav** as part of the MSc (Data Analytics) curriculum, Pillai College of Arts, Commerce & Science (Autonomous), New Panvel — under the guidance of **Prof. Omkar Sherkhane**.

---

## ⚠️ Disclaimer

This project was built for academic purposes to demonstrate real-time computer vision and deep learning techniques. It is a prototype and has not been validated for deployment as a safety-critical system in actual vehicles.
