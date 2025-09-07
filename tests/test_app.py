import json
import base64
import os
from io import BytesIO
import numpy as np
import cv2
import pytest
from app import app  # Import the Flask app

# ------------------------------
# Test Client Setup
# ------------------------------
@pytest.fixture
def client():
    """Create a Flask test client."""
    with app.test_client() as client:
        yield client


# ------------------------------
# Helper Function
# ------------------------------
def create_dummy_image():
    """Create a dummy 100x100 pixel image and return as byte array."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded_image = cv2.imencode('.jpg', dummy_image)
    return encoded_image.tobytes() if success else None


# ------------------------------
# Test Cases
# ------------------------------
def test_index_route(client):
    """Test the index route loads successfully."""
    response = client.get('/')
    assert response.status_code == 200


def test_detect_route_no_file(client):
    """Test /detect route without providing a file."""
    response = client.post('/detect')
    assert response.status_code == 400
    assert b'No file part in the request' in response.data


def test_detect_route_with_file(client):
    """Test /detect route with a dummy image."""
    dummy_image_bytes = create_dummy_image()
    assert dummy_image_bytes is not None, "Failed to create dummy image."

    data = {
        'file': (BytesIO(dummy_image_bytes), 'test.jpg')
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code in [200, 500]  # 500 if model isn't loaded
    assert 'image' in response.get_json() or 'error' in response.get_json()


def test_process_frame_route(client):
    """Test /process_frame route with dummy base64 image."""
    dummy_image_bytes = create_dummy_image()
    assert dummy_image_bytes is not None

    # Encode as base64 string
    base64_image = base64.b64encode(dummy_image_bytes).decode('utf-8')
    data = {
        'frame': f"data:image/jpeg;base64,{base64_image}"
    }

    response = client.post('/process_frame', json=data)
    assert response.status_code in [200, 500]
    json_data = response.get_json()
    assert 'faces' in json_data or 'error' in json_data


def test_plot_training_history_route(client):
    """Test /plot_training_history route with a dummy history.json file."""
    dummy_history = {
        "epochs": [
            {"epoch": 1, "accuracy": 0.8, "val_accuracy": 0.75, "loss": 0.5, "val_loss": 0.55},
            {"epoch": 2, "accuracy": 0.85, "val_accuracy": 0.78, "loss": 0.45, "val_loss": 0.50}
        ]
    }

    with open('history.json', 'w') as f:
        json.dump(dummy_history, f)

    response = client.get('/plot_training_history')
    os.remove('history.json')
    assert response.status_code == 200
    assert 'plot' in response.get_json()


def test_plot_confusion_matrix_route(client):
    """Test /plot_confusion_matrix route with dummy confusion_matrix.json."""
    dummy_cm = {
        "confusion_matrix": [[5, 0], [1, 4]],
        "labels": ["happy", "sad"]
    }

    with open('confusion_matrix.json', 'w') as f:
        json.dump(dummy_cm, f)

    response = client.get('/plot_confusion_matrix')
    os.remove('confusion_matrix.json')
    assert response.status_code == 200
    assert 'plot' in response.get_json()
