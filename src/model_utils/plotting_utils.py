import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os

def plot_to_result(image,
                   annotations,
                   bboxes=None,
                   points=None,
                   point_label=None,
                   mask_random_color=True,
                   better_quality=True,
                   retina=False,
                   withContours=True,
                   device: str = "cuda") -> np.ndarray:
    
    if isinstance(annotations[0], dict):
        annotations = [annotation['segmentation'] for annotation in annotations]
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    original_h = image.shape[0]
    original_w = image.shape[1]
    if sys.platform == "darwin":
        plt.switch_backend("TkAgg")
    plt.figure(figsize=(original_w / 100, original_h / 100))
    # Add subplot with no margin.
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.imshow(image)
    if better_quality:
        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    if device == 'cpu':
        annotations = np.array(annotations)
        fast_show_mask(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bboxes=bboxes,
            points=points,
            pointlabel=point_label,
            retinamask=retina,
            target_height=original_h,
            target_width=original_w,
        )
    else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
            fast_show_mask_gpu(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bboxes=bboxes,
            points=points,
            pointlabel=point_label,
            retinamask=retina,
            target_height=original_h,
            target_width=original_w,
        )
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()
    if withContours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']
            annotation = mask.astype(np.uint8)
            if not retina:
                annotation = cv2.resize(
                    annotation,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)
        plt.imshow(contour_mask)

    plt.axis('off')
    fig = plt.gcf()
    plt.draw()

    try:
        buf = fig.canvas.tostring_rgb()
    except AttributeError:
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
    cols, rows = fig.canvas.get_width_height()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
    result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    plt.close()
    result = result[:, :, ::-1] #convert back to RGB
    return result
        
def store(result, 
          output_path
          ) -> None:

    path = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(output_path, result)
    
#   CPU post process
def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bboxes=None,
    points=None,
    pointlabel=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    #Sort annotations based on area.
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color:
        color = np.random.random((msak_sum, 1, 1, 3))
    else:
        color = np.ones((msak_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
    transparency = np.ones((msak_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    show = np.zeros((height, weight, 4))
    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # Use vectorized indexing to update the values of 'show'.
    show[h_indices, w_indices, :] = mask_image[indices]
    if bboxes is not None:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
    # draw point
    if points is not None:
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
            s=20,
            c='y',
        )
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
            s=20,
            c='m',
        )

    if not retinamask:
        show = cv2.resize(show, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    ax.imshow(show)

def fast_show_mask_gpu(
    annotation,
    ax,
    random_color=False,
    bboxes=None,
    points=None,
    pointlabel=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    # Find the index of the first non-zero value at each position.
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color:
        color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
    else:
        color = torch.ones((msak_sum, 1, 1, 3)).to(annotation.device) * torch.tensor([
            30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
    transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    # Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
    show = torch.zeros((height, weight, 4)).to(annotation.device)
    try:
        h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
    except:
        h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # Use vectorized indexing to update the values of 'show'.
    show[h_indices, w_indices, :] = mask_image[indices]
    show_cpu = show.cpu().numpy()
    if bboxes is not None:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
    # draw point
    if points is not None:
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
            s=20,
            c='y',
        )
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
            s=20,
            c='m',
        )
    if not retinamask:
        show_cpu = cv2.resize(show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    ax.imshow(show_cpu)