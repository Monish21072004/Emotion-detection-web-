import os
os.environ["FLASK_TESTING"] = "1"

from io import BytesIO
import pytest
from app import app



@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """
    Test that the index route returns a 200 status code and
    contains expected HTML content.
    """
    response = client.get('/')
    assert response.status_code == 200
    # Check that the response contains HTML content
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
    Expect an error response.
    """
    invalid_frame = (
        "data:image/jpeg;base64,not_base64_encoded_data"
    )
    response = client.post(
        '/process_frame', json={'frame': invalid_frame}
    )
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
    data = {
        'file': (BytesIO(b'not an image content'), 'test.txt')
    }
    response = client.post(
        '/detect', data=data, content_type='multipart/form-data'
    )
    assert response.status_code in [400, 500]
    data = response.get_json()
    assert 'error' in data


def test_plot_training_history_file_not_found(client):
    """
    Test the /plot_training_history endpoint when history.json is missing.
    Expects an error response.
    """
    response = client.get('/plot_training_history')
    assert response.status_code in [400, 404]
    data = response.get_json()
    assert 'error' in data


def test_plot_confusion_matrix_file_not_found(client):
    """
    Test the /plot_confusion_matrix endpoint when confusion_matrix.json is missing.
    Expects an error response.
    """
    response = client.get('/plot_confusion_matrix')
    assert response.status_code in [400, 404]
    data = response.get_json()
    assert 'error' in data
