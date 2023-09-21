import pytest
import requests

INFER_API_URL = "http://127.0.0.1:4000/infer"

def test_extra_fields():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "text", 
            "text_prompt": "dogs", 
            "point_prompt": "[[0.25, 0.25], [0.5, 0.5]]", 
            "point_label": "[1, 0]"
            }
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code != 200


def test_mismatched_labels():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "points", 
            "point_prompt": "[[0.25, 0.25], [0.5, 0.5]]", 
            "point_label": "[1, 0, 1]"
            }
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code != 200


def test_points_in_range():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "points", "point_prompt": "[[1.5, 100], [-0.5, 0.5]]", "point_label": "[1, 0]"}
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code != 200


def test_irrelevant_fields():
    files={"image": open("./assets/dogs.jpg", "rb")}
    data = {"mode": "text", 
            "text_prompt": "yellow dog", 
            "foo": "bar", 
            "bar": "foo"
            }
    response = requests.post(INFER_API_URL, files=files, data=data)
    
    assert response.status_code == 200




