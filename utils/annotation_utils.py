import torch
from PIL import Image
import cv2
import numpy as np

try:
    import clip  # for linear_assignment

except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip


def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

@torch.no_grad()
def retrieve(model, preprocess, elements, search_text: str, device) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def crop_image(image, format_results):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ori_w, ori_h = image.size
    annotations = format_results
    mask_h, mask_w = annotations[0]['segmentation'].shape
    if ori_w != mask_w or ori_h != mask_h:
        image = image.resize((mask_w, mask_h))
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    filter_id = []
    # annotations, _ = filter_masks(annotations)
    # filter_id = list(_)
    for _, mask in enumerate(annotations):
        if np.sum(mask['segmentation']) <= 100:
            filter_id.append(_)
            continue
        bbox = _get_bbox_from_mask(mask['segmentation'])  # mask 的 bbox
        cropped_boxes.append(_segment_image(image, bbox))  
        # cropped_boxes.append(segment_image(image,mask["segmentation"]))
        cropped_images.append(bbox)  # Save the bounding box of the cropped image.

    return cropped_boxes, cropped_images, not_crop, filter_id, annotations

def _get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            # Merge multiple bounding boxes into one.
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]

def _segment_image(image, bbox):
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new('RGB', image.size, (255, 255, 255))
    # transparency_mask = np.zeros_like((), dtype=np.uint8)
    transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image