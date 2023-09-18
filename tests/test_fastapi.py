import pytest
import requests_mock
import requests

API_URL = "http://127.0.0.1:4000/ping/"

CORRECT_RESPONSE = {"message": "pong"}

def test_ping_api(requests_mock):

    requests_mock.get(API_URL, json=CORRECT_RESPONSE, status_code=200)
    response = requests.get(API_URL)
        
    assert response.json() == CORRECT_RESPONSE



