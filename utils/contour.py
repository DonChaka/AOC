from copy import copy
from os import listdir
from os.path import join
from typing import Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import jaccard_score

from utils.mask import __histogram_equalization, get_mask_dict


def __blur_masks(masks: dict, kernel: int = 7) -> dict:
    kernel_size = (kernel, kernel)
    result = {}

    for key, val in masks.items():
        result[key] = cv2.GaussianBlur(val, kernel_size, 0)

    return result


def __sharpen_masks(masks: dict, kernel: int = 5) -> dict:
    result = {}

    for key, val in masks.items():
        result[key] = cv2.convertScaleAbs(cv2.Laplacian(val, cv2.CV_8U, kernel))

    return result


def __get_segmented_images(img: np.ndarray, masks: dict) -> dict:
    return {
        'red': cv2.bitwise_and(img, img, mask=masks['red']),
        'blue': cv2.bitwise_and(img, img, mask=masks['blue']),
        'yellow': cv2.bitwise_and(img, img, mask=masks['yellow'])
    }


def __get_edges_per_channel(seg_images: dict, low_thresh: int = 50, high_thresh: int = 150) -> dict:
    return {
        'red': cv2.Canny(seg_images['red'], low_thresh, high_thresh),
        'blue': cv2.Canny(seg_images['blue'], low_thresh, high_thresh),
        'yellow': cv2.Canny(seg_images['yellow'], low_thresh, high_thresh)
    }


def __find_contours(edges: np.ndarray) -> list:
    red_contours, red_hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [red_contours[i] for i, element in enumerate(red_hierarchy[0]) if element[3] == -1]


def __find_contours_per_channel(all_edges: dict) -> dict:
    return {
        'red': __find_contours(all_edges['red']),
        'blue': __find_contours(all_edges['blue']),
        'yellow': __find_contours(all_edges['yellow'])
    }


def __if_valid_react(width: int, high: int, source_image_area: float) -> bool:
    min_area = source_image_area * 0.001
    rect_area = width * high
    ratio = width / high

    return 0.5 < ratio < 1.2 and min_area < rect_area


def __get_rects_per_one_channel(contours: np.ndarray, image_area: float) -> tuple:
    result = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if __if_valid_react(width=w, high=h, source_image_area=image_area):
            result.append((x, y, w, h))

    return tuple(result)


def __get_all_rects(contours: dict, image_area: float) -> tuple:
    return (
        *__get_rects_per_one_channel(contours=contours['red'], image_area=image_area),
        *__get_rects_per_one_channel(contours=contours['blue'], image_area=image_area),
        *__get_rects_per_one_channel(contours=contours['yellow'], image_area=image_area)
    )


def add_rects_on_image(
        img: np.ndarray, all_rects: tuple, rect_thickness: float = 2, rect_color: Union[tuple, int] = (255, 255, 0)
) -> None:
    for rect in all_rects:
        cv2.rectangle(
            img,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            rect_color,
            rect_thickness
        )


def match_rects_to_images(image: np.ndarray) -> tuple:
    equ_hsv_image = rgb2hsv(__histogram_equalization(data=image))
    masks = __sharpen_masks(masks=__blur_masks(masks=get_mask_dict(img=equ_hsv_image)))

    seg_images = __get_segmented_images(img=image, masks=masks)
    edges = __get_edges_per_channel(seg_images=seg_images)

    return __get_all_rects(
        contours=__find_contours_per_channel(edges),
        image_area=image.shape[0] * image.shape[1]
    )


def get_rects_for_images(home: str, source_image_shape: tuple) -> tuple:
    images_data = []
    images_append = images_data.append

    for file_name in listdir(home):
        image = resize(
            image=imread(join(home, file_name)),
            output_shape=source_image_shape,
            preserve_range=True
        ).astype(np.uint8)

        images_append({
            'file_name': file_name,
            'img': image,
            'rects': match_rects_to_images(image=image)
        })

    return tuple(images_data)


def add_rects_on_images(images_with_rects_data: tuple) -> tuple:
    result = []
    result_append = result.append

    for img_data in images_with_rects_data:
        image = copy(img_data['img'])

        add_rects_on_image(
            img=image,
            all_rects=img_data['rects']
        )

        result_append({'file': img_data['file_name'], 'img': image})

    return tuple(result)


def create_rect_mask_for_images(images_with_rects_data: tuple) -> tuple:
    result = []
    result_append = result.append

    for img_data in images_with_rects_data:
        image = np.ones_like(img_data['img'])

        add_rects_on_image(
            img=image,
            all_rects=img_data['rects'],
            rect_color=0,
            rect_thickness=-1
        )

        result_append({'file': img_data['file_name'], 'img': image})

    return tuple(result)


def calculate_jaccard_score(home_path: str, source_image_shape: tuple, detected_rects: list) -> float:
    results = []
    results_append = results.append

    for obj in detected_rects:
        test_img = np.where(resize(
            image=imread(join(home_path, obj['file']), as_gray=True),
            output_shape=source_image_shape,
            preserve_range=True
        ).astype(np.uint8) < 0.5, 0, 1)

        results_append(jaccard_score(
            y_true=test_img.ravel(),
            y_pred=obj['img'].ravel()
        ))

    return np.array(results).mean()


def plot_images(images_data: list) -> None:
    fig, axes = plt.subplots(ncols=2, nrows=len(images_data) // 2 + 1, figsize=(24, 25))
    axes = axes.ravel()

    for i, img_data in enumerate(images_data):
        axes[i].imshow(img_data['img'], cmap=plt.cm.brg)
        axes[i].set_title(img_data['file'])
        axes[i].axis('off')

    for i in range(len(images_data), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
