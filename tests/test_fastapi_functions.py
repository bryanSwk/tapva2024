import pytest
import requests

INFER_API_URL = "http://127.0.0.1:4000/infer/"

def test_infer_everything():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "everything"}
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code == 200
    
    content_type = response.headers.get("Content-Type")
    assert content_type.startswith("image/")
    
    assert len(response.content) > 0
    

def test_infer_text():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "text", "text_prompt": "yellow dog"}
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code == 200
    
    content_type = response.headers.get("Content-Type")
    assert content_type.startswith("image/")
    
    assert len(response.content) > 0


def test_infer_box():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "box", "box_prompt": "[[0.5, 0.5, 0.5, 0.5]]"}
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code == 200
    
    content_type = response.headers.get("Content-Type")
    assert content_type.startswith("image/")
    
    assert len(response.content) > 0


def test_infer_points():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "points", "point_prompt": "[[0.25, 0.25], [0.5, 0.5]]", "point_label": "[1, 0]"}
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code == 200
    
    content_type = response.headers.get("Content-Type")
    assert content_type.startswith("image/")
    
    assert len(response.content) > 0
