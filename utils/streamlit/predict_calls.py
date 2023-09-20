import requests
import json

class PredictAPI:
    def __init__(self) -> None:
        self.url = 'http://127.0.0.1:4000/infer'
        self.headers = {
            'accept': 'application/json',
        }
        self.error_msg = "Response"

    def predict_everything(self, image):
        files={"image": image}
        mode = "everything"
        data = {"mode": mode}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        if response.status_code == 200:
            return response
        else:
            return "Prediction Error"
    def predict_text(self, image, text):
        files={"image": image}
        mode = 'text'
        data = {"mode": mode, "text_prompt":f"{text}"}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        if response.status_code == 200:
            return response
        else:
            return "Prediction Error"

    def predict_box(self, image, bboxes):
        files={"image": image}
        mode = "box"
        data = {"mode": mode, "box_prompt":f"{bboxes}"}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        if response.status_code == 200:
            return response
        else:
            return "Prediction Error"
    def predict_points(self, image, points, pointlabels):
        files={"image": image}
        mode = "points"
        data = {"mode": mode, "point_prompt": f"{points}", "point_label": f"{pointlabels}"}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        if response.status_code == 200:
            return response
        else:
            return "Prediction Error"
