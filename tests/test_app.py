import json
import base64
from io import BytesIO
import pytest
import numpy as np
import cv2

# Import your Flask app
from app import app

# Create a test client using pytest fixture
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """
    Test that the index route returns a 200 status code and contains expected HTML content.
    """
    response = client.get('/')
    assert response.status_code == 200
    # Check that the response likely contains HTML (could be modified to match your actual template)
    assert b'<html' in response.data or b'<!DOCTYPE html>' in response.data

def test_process_frame_without_data(client):
    """
    Test the /process_frame endpoint when no frame data is provided.
    It should return a 400 error with an error message.
    """
    response = client.post('/process_frame', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_process_frame_with_invalid_image(client):
    """
    Test the /process_frame endpoint with an invalid base64 encoded string.
    Expect an error response (status code 400 or 500).
    """
    # Provide an improperly formatted frame value
    invalid_frame = "data:image/jpeg;base64,not_base64_encoded_data"
    response = client.post('/process_frame', json={'frame': invalid_frame})
    # The response could be 400 or 500 depending on how the exception is handled
    assert response.status_code in [400, 500]
    data = response.get_json()
    assert 'error' in data

def test_detect_without_file(client):
    """
    Test the /detect endpoint when no file is provided.
    It should return a 400 error and an error message.
    """
    response = client.post('/detect')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_detect_with_invalid_file(client):
    """
    Test the /detect endpoint with a file that does not contain a valid image.
    The endpoint should return an error status code.
    """
    # Create a dummy file with non-image content
    data = {
        'file': (BytesIO(b'not an image content'), 'test.txt')
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    # Expecting an error because the file cannot be processed as an image
    assert response.status_code in [400, 500]
    data = response.get_json()
    assert 'error' in data

def test_plot_training_history_file_not_found(client):
    """
    Test the /plot_training_history endpoint when history.json is missing.
    Expects an error response.
    """
    response = client.get('/plot_training_history')
    # Depending on the file existence check, we expect a 404 or a 400 error
    assert response.status_code in [400, 404]
    data = response.get_json()
    assert 'error' in data

def test_plot_confusion_matrix_file_not_found(client):
    """
    Test the /plot_confusion_matrix endpoint when confusion_matrix.json is missing.
    Expects an error response.
    """
    response = client.get('/plot_confusion_matrix')
    # Expecting a 404 (or sometimes 400) if the file is not found
    assert response.status_code in [400, 404]
    data = response.get_json()
    assert 'error' in data
