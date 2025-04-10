import os
import sys
import pytest
import json

# Add the directory containing app.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Import the Flask app

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the index route."""
    response = client.get('/')
    assert response.status_code == 200

def test_process_frame_no_data(client):
    """Test /process_frame with no data."""
    response = client.post('/process_frame', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_detect_no_file(client):
    """Test /detect with no file uploaded."""
    response = client.post('/detect')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
