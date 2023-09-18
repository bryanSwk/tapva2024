import requests
import json

class PredictAPI:
    def __init__(self) -> None:
        self.url = 'http://127.0.0.1:4000/infer/'
        self.headers = {
            'accept': 'application/json',
        }

    def predict_everything(self, image):
        files={"image": image}
        mode = "everything"
        data = {'request': f'{{"mode":"{mode}"}}'}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        return response
    def predict_text(self, image, text):
        files={"image": image}
        mode = 'text'
        data = {'request': f'{{"mode":"{mode}", "text_prompt":"{text}"}}'}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        return response

    def predict_box(self, image, bboxes):
        files={"image": image}
        mode = "box"
        data = {"request": json.dumps({"mode": f"{mode}", "box_prompt": bboxes})}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        return response
    
    def predict_points(self, image, points, pointlabels):
        files={"image": image}
        mode = "points"
        data = {"request": json.dumps({"mode": f"{mode}", "point_prompt": points, "point_label": pointlabels})}
        response = requests.post(self.url, headers=self.headers, files=files, data=data)
        return response
