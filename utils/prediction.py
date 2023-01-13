from os.path import join

import cv2
import numpy as np
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.transform import resize
from utils.classifier import get_hog_descriptors
from utils.colours import COLOR


def add_rects_with_pred_on_image(
        image: np.ndarray, rects_data: tuple, model: object, slice_image_shape: tuple, rect_thickness: int,
        font_size: int, font_thickness: int, hog_params: dict
) -> None:
    for rect in rects_data:
        sign = resize(
            image=image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]],
            output_shape=slice_image_shape,
            preserve_range=True
        ).astype(np.uint8)

        a = get_hog_descriptors(images=[rgb2hsv(sign)], hog_params=hog_params)
        pred = model.predict(a)

        cv2.rectangle(
            image,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            COLOR['yellow'],
            rect_thickness
        )

        cv2.putText(
            image,
            pred[0],
            (rect[0] + rect[2], rect[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            COLOR['yellow'],
            font_thickness,
            cv2.LINE_AA
        )


def add_rects_with_pred_on_images(
        rects_data: dict, model: object, source_image_shape: tuple, home_path: str, slice_image_shape: tuple,
        rect_thickness: int, font_size: int, font_thickness: int, hog_params: dict
) -> list:
    result = []
    result_append = result.append

    for i, rects_dict in enumerate(rects_data):
        add_rects_with_pred_on_image(
            image=rects_dict['img'],
            rects_data=rects_dict['rects'],
            model=model,
            slice_image_shape=slice_image_shape,
            rect_thickness=rect_thickness,
            font_size=font_size,
            font_thickness=font_thickness,
            hog_params=hog_params
        )

        result_append({'file': rects_dict['file_name'], 'img': rects_dict['img']})

    return result
