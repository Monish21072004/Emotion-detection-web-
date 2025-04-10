import sys
import os
# Ensure the repository root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app import app  # Now this import should work correctly

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200

def test_process_frame_no_data(client):
    response = client.post('/process_frame', json={})
    assert response.status_code == 400
    assert b'No frame data provided' in response.data

def test_detect_emotion_no_file(client):
    response = client.post('/detect', data={})
    assert response.status_code == 400
    assert b'No file part in the request' in response.data

def test_plot_training_history_missing_file(client):
    response = client.get('/plot_training_history')
    assert response.status_code in [200, 404]  # file may not exist

def test_plot_confusion_matrix_missing_file(client):
    response = client.get('/plot_confusion_matrix')
    assert response.status_code in [200, 404]  # file may not exist
