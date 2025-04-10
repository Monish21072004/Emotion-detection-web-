import os
import json
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set image dimensions and emotion labels (must match your training)
IMG_SIZE = 48
EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load the trained model (adjust the model path if needed)
MODEL_PATH = 'emotion-detect.keras'
emotion_model = load_model(MODEL_PATH)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    """Resize, normalize, and reshape a face image for prediction."""
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_normalized = face_resized / 255.0
    face_reshaped = np.reshape(face_normalized, (1, IMG_SIZE, IMG_SIZE, 1))
    return face_reshaped

@app.route('/')
def index():
    return render_template('trial.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        if 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400

        header, encoded = data['frame'].split(",", 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        faces_data = []
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_input = preprocess_face(face_roi)
            prediction = emotion_model.predict(face_input)
            max_index = int(np.argmax(prediction))
            emotion = EMOTION_LABELS[max_index]
            faces_data.append({
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'emotion': emotion
            })

        return jsonify({'faces': faces_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Could not process the image'}), 400

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    detected_emotions = []

    for (x, y, w, h) in faces:
        face_roi = gray_image[y:y + h, x:x + w]
        face_input = preprocess_face(face_roi)
        prediction = emotion_model.predict(face_input)
        max_index = int(np.argmax(prediction))
        emotion = EMOTION_LABELS[max_index]
        detected_emotions.append(emotion)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    emotion_result = ', '.join(detected_emotions) if detected_emotions else 'No face detected'

    return jsonify({'image': jpg_as_text, 'emotion': emotion_result})

@app.route('/plot_training_history', methods=['GET'])
def plot_training_history():
    """
    Endpoint to plot the training history.
    Loads actual training history from history.json (using the "epochs" key)
    and returns a base64-encoded plot image.
    """
    try:
        if not os.path.exists('history.json'):
            return jsonify({'error': 'Training history file not found.'}), 404

        with open('history.json', 'r') as f:
            history = json.load(f)

        # Check for the "epochs" key and extract data
        if "epochs" not in history:
            return jsonify({'error': 'Training history file structure is incorrect. Expected key "epochs".'}), 400

        epoch_data = history["epochs"]
        if not epoch_data:
            return jsonify({'error': 'No epoch data found in training history.'}), 400

        epochs = [entry["epoch"] for entry in epoch_data]
        train_acc = [entry["accuracy"] for entry in epoch_data]
        val_acc = [entry["val_accuracy"] for entry in epoch_data]
        train_loss = [entry["loss"] for entry in epoch_data]
        val_loss = [entry["val_loss"] for entry in epoch_data]

        plt.figure(figsize=(12, 5))
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({'plot': plot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot_confusion_matrix', methods=['GET'])
def plot_confusion_matrix_endpoint():
    """
    Endpoint to plot the confusion matrix.
    Loads the confusion matrix from confusion_matrix.json and returns a base64-encoded plot image.
    """
    try:
        if not os.path.exists('confusion_matrix.json'):
            return jsonify({'error': 'Confusion matrix file not found.'}), 404

        with open('confusion_matrix.json', 'r') as f:
            cm_data = json.load(f)
        cm = np.array(cm_data['confusion_matrix'])
        labels = cm_data.get('labels', EMOTION_LABELS)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({'plot': plot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app; remove debug=True in production
    app.run(host='0.0.0.0', port=5000, debug=True)
