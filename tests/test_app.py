import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app  # Ensure that your Flask app is imported from your project file

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """
    Test the index endpoint to check if it returns status code 200.
    """
    response = client.get('/')
    assert response.status_code == 200

def test_process_frame_no_data(client):
    """
    Test /process_frame when no data is provided.
    Expected: Returns a 400 error with an error message.
    """
    response = client.post('/process_frame', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_detect_no_file(client):
    """
    Test /detect endpoint when no file is uploaded.
    Expected: Returns a 400 error with an error message.
    """
    response = client.post('/detect')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
