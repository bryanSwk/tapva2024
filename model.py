from typing import List
import logging
import os
import numpy as np

from fastsam import FastSAM, FastSAMPrompt
import torch
import cv2
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

from utils.annotation_utils import format_results, retrieve, crop_image
from utils.plotting_utils import store, plot_to_result


try:
    import clip  # for linear_assignment

except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)



class InferenceModel:
    def __init__(self, config: DictConfig) -> None:
        
        self.cfg = config
        self.model = FastSAM(config.inference.weights_path)
        self.device = torch.device("cuda"
                                    if torch.cuda.is_available()
                                    else "mps"
                                    if torch.backends.mps.is_available()
                                    else "cpu"
                                    )
        #done to speed up subsequent inference.
        self.model(config.inference.test_image,
                   device=self.device,
                   retina_masks=True,
                   imgsz=1024,
                   conf=0.4,
                   iou=0.9,
                   )
        
        logging.info(f"Successfully loaded FastSAM from {config.inference.weights_path} to device: {self.device}.")

        self.clip_model, self.preprocess = clip.load(config.inference.clip_model, device=self.device)

        logging.info(f"Loaded CLIP Model: {config.inference.clip_model} to device: {self.device}.")

        self.function_mapping = {'everything': self.annotate_everything,
                                'text': self.annotate_text,
                                'box': self.annotate_box,
                                'points': self.annotate_point
        }
        
    def hotreload_model(self, weights_path: str) -> None:
        self.model = FastSAM(weights_path)

    def predict(self, image, payload):
        if payload['data']['mode'] in self.function_mapping:
            func_kwargs = {key: payload['data'][key] for key in payload['data'] if key != 'mode'}
            annotation = self.function_mapping[payload['data']['mode']](image, **func_kwargs)
            print(payload['data']['mode'])
            print(annotation)
            if payload['data']['mode'] == "everything":
                annotation = annotation[0].masks.data
            result = plot_to_result(image, annotation)
            output_path = os.path.join(self.cfg.inference.output_path, 'output.jpg')
            store(result, output_path)
            print("success!")
        
        else:
            print("wrong format")


    def annotate_everything(self, image):
        image.show()
        ann = self.model(image,
                        device=self.device,
                        retina_masks=True,
                        imgsz=1024,
                        conf=0.4,
                        iou=0.9,
                        )
        return ann
    
    def annotate_box(self, image, bboxes: List[List[int]]):
        ann = self.annotate_everything(image)
        if ann and bboxes is not None:
            max_iou_index = []
            for bbox in bboxes:
                assert (bbox[2] != 0 and bbox[3] != 0)
                masks = ann[0].masks.data
                target_height = np.array(image).shape[0]
                target_width = np.array(image).shape[1]
                # Extra step to undo normalization
                bbox[0] = int(bbox[0] * target_width)
                bbox[1] = int(bbox[1] * target_height)
                bbox[2] = int(bbox[2] * target_width)
                bbox[3] = int(bbox[3] * target_height)
                h = masks.shape[1]
                w = masks.shape[2]
                if h != target_height or w != target_width:
                    bbox = [
                        int(bbox[0] * w / target_width),
                        int(bbox[1] * h / target_height),
                        int(bbox[2] * w / target_width),
                        int(bbox[3] * h / target_height), ]
                bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
                bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
                bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
                bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

                # IoUs = torch.zeros(len(masks), dtype=torch.float32)
                bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

                masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
                orig_masks_area = torch.sum(masks, dim=(1, 2))

                union = bbox_area + orig_masks_area - masks_area
                IoUs = masks_area / union
                max_iou_index.append(int(torch.argmax(IoUs)))
            max_iou_index = list(set(max_iou_index))
            return np.array(masks[max_iou_index].cpu().numpy())
        
        else:
            return []
    
    def annotate_text(self, image, text: str):
        ann = self.annotate_everything(image)
        if ann and text is not None:
            results = format_results(ann[0], 0)
            cropped_boxes, cropped_images, not_crop, filter_id, annotations = crop_image(image, results)
            scores = retrieve(self.clip_model, self.preprocess, cropped_boxes, text, device=self.device)
            max_idx = scores.argsort()
            max_idx = max_idx[-1]
            max_idx += sum(np.array(filter_id) <= int(max_idx))

            return np.array([annotations[max_idx]['segmentation']])
        
        else:
            print("NONE")
            return []

    def annotate_point(self, image, points: List[List[int]], pointlabel: List):
        ann = self.annotate_everything(image)
        if ann and points and pointlabel is not None:
            masks = format_results(ann[0], 0)
            target_height = np.array(image).shape[0]
            target_width = np.array(image).shape[1]
            points = [[int(point[0] * target_width), int(point[1] * target_height)] for point in points]
            h = masks[0]['segmentation'].shape[0]
            w = masks[0]['segmentation'].shape[1]
            if h != target_height or w != target_width:
                points = [[int(point[0] * w / target_width), int(point[1] * h / target_height)] for point in points]
            onemask = np.zeros((h, w))
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            for i, annotation in enumerate(masks):
                if type(annotation) == dict:
                    mask = annotation['segmentation']
                else:
                    mask = annotation
                for i, point in enumerate(points):
                    if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                        onemask[mask] = 1
                    if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                        onemask[mask] = 0
            onemask = onemask >= 1
            return np.array([onemask])
        
    

        

@hydra.main(config_path="./config", config_name="config.yaml")
def run_standalone(config):
    model = InferenceModel(config)
    # model.predict()


if __name__ == "__main__":
    run_standalone()