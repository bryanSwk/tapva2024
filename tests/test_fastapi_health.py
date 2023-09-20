import pytest
import requests

PING_API_URL = "http://127.0.0.1:4000/ping/"
INFER_API_URL = "http://127.0.0.1:4000/infer/"


def test_ping_api():
    CORRECT_RESPONSE = {"message": "pong"}

    response = requests.get(PING_API_URL)
        
    assert response.json() == CORRECT_RESPONSE


def test_infer_everything():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "everything"}
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code == 200
