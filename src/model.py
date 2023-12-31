from typing import List
import logging
import numpy as np

import torch

from omegaconf import DictConfig

from src.fastsam import FastSAM
from src.model_utils.annotation_utils import format_results, retrieve, crop_image
from src.model_utils.plotting_utils import store, plot_to_result

try:
    import clip  # for linear_assignment

except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class InferenceModel:
    """
    Wrapper for an Inference Model.

    This class provides a wrapper for an inference model, making it easier to perform inference
    tasks with the FastSAM model. It includes methods for performing inference on input data.

    Args:
        config (DictConfig): A hydra configuration object containing model and inference settings.
    """

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

    def predict(self, image, payload: dict):
        """
        Perform a prediction using the provided PIL Image and payload.

        This method takes a PIL Image as the input image and an additional payload dictionary,
        and uses them to make a prediction.

        Args:
            image (PIL.Image): The input image for prediction in the form of a PIL Image.
            payload (dict): Additional data or information required for the prediction.
                    The content and format of the payload may vary based on the inference mode.

        Returns:
            PIL.Image: The prediction result, returned as a PIL Image.


        """
        if payload['mode'] in self.function_mapping:
            func_kwargs = {key: payload[key] for key in payload if key != 'mode'}
            annotation = self.function_mapping[payload['mode']](image, **func_kwargs)
            if payload['mode'] == "everything":
                annotation = annotation[0].masks.data
            result = plot_to_result(image, annotation)
            # Uncomment for local storage
            # output_path = os.path.join(self.cfg.inference.output_path, 'output.jpg')
            # store(result, output_path) 
            return result
        
        else:
            raise ValueError("Mode not supported. Try: 'everything', 'text', 'box', 'points'.")


    def annotate_everything(self, image):
        """
        Annotate with FastSAM using everything mode for the provided image.

        Args:
            image (PIL.Image): The input image for annotation.

        Returns:
            Annotation: Annotation result for EverythingMode.
        """
        ann = self.model(image,
                        device=self.device,
                        retina_masks=True,
                        imgsz=1024,
                        conf=0.4,
                        iou=0.9,
                        )
        return ann
    
    def annotate_box(self, image, box_prompt: List[List[int]]):
        """
        Annotate with FastSAM using BoxMode for the provided image.

        Args:
            image (PIL.Image): The input image for annotation.
            box_prompt (List[List[int]]): The bounding box coordinates.

        Returns:
            Annotation: Annotation result for BoxMode.
        """
        ann = self.annotate_everything(image)
        if ann and box_prompt is not None:
            max_iou_index = []
            for bbox in box_prompt:
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
    
    def annotate_text(self, image, text_prompt: str):
        """
        Annotate with FastSAM using TextMode for the provided image.

        Args:
            image (PIL.Image): The input image for annotation.
            text_prompt (str): A text prompt of any valid string.

        Returns:
            Annotation: Annotation result for TextMode.
        """
        ann = self.annotate_everything(image)
        if ann and text_prompt is not None:
            results = format_results(ann[0], 0)
            cropped_boxes, cropped_images, not_crop, filter_id, annotations = crop_image(image, results)
            scores = retrieve(self.clip_model, self.preprocess, cropped_boxes, text_prompt, device=self.device)
            max_idx = scores.argsort()
            max_idx = max_idx[-1]
            max_idx += sum(np.array(filter_id) <= int(max_idx))

            return np.array([annotations[max_idx]['segmentation']])
        
        else:
            return []

    def annotate_point(self, image, point_prompt: List[List[int]], point_label: List):
        """
        Annotate with FastSAM using PointMode for the provided image.

        Args:
            image (PIL.Image): The input image for annotation.
            point_prompt (List[List[int]]): A list containing normalized point coordinates.
            point_label (List): Corresponding list containing label information for points.

        Returns:
            Annotation: Annotation result for PointMode.
        """
        ann = self.annotate_everything(image)
        if ann and point_prompt and point_label is not None:
            masks = format_results(ann[0], 0)
            target_height = np.array(image).shape[0]
            target_width = np.array(image).shape[1]
            points = [[int(point[0] * target_width), int(point[1] * target_height)] for point in point_prompt]
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
                    if mask[point[1], point[0]] == 1 and point_label[i] == 1:
                        onemask[mask] = 1
                    if mask[point[1], point[0]] == 1 and point_label[i] == 0:
                        onemask[mask] = 0
            onemask = onemask >= 1
            return np.array([onemask])
