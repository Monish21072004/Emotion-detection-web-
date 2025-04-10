# Real-Time Emotion Detection Web App

A deep learning-based web application that detects and classifies human emotions from facial expressions in real time using OpenCV and TensorFlow. The app supports both image uploads and webcam input, and includes visual tools like training history and a confusion matrix for evaluation.

---

## 📌 Features

- 🎯 Real-time emotion detection from webcam feed
- 🖼️ Emotion detection from uploaded images
- 🧠 CNN-based emotion classifier trained on facial expression data
- 📊 Interactive training history and confusion matrix plots
- 🖥️ Web-based interface built with Flask
- 💬 Supports 7 emotions: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, and `surprised`

---

## 🧠 Emotion Labels

| Label Index | Emotion     |
|-------------|-------------|
| 0           | Angry       |
| 1           | Disgusted   |
| 2           | Fearful     |
| 3           | Happy       |
| 4           | Neutral     |
| 5           | Sad         |
| 6           | Surprised   |

---

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib & Seaborn
- HTML / JS (for front-end webcam integration)

---

## 🚀 Getting Started

### 🔧 Prerequisites

Make sure you have the following installed:
- Python 3.7+
- pip

Install required libraries:
```bash
pip install -r requirements.txt
emotion-detection-app/
│
├── app.py                      # Flask backend
├── emotion-detect.keras        # Pre-trained CNN model
├── history.json                # Training history (accuracy/loss per epoch)
├── confusion_matrix.json       # Confusion matrix data
├── templates/
│   └── trial.html              # Front-end HTML page
└── uploads/                    # Folder to store uploaded images
python app.py
http://localhost:5000

