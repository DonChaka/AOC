from os.path import join

import cv2
import numpy as np
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.transform import resize

from utis.classifier import get_hog_descriptors
from utis.colours import COLOR


def add_rects_with_pred_on_image(
        image: np.ndarray, color_rects_data: dict, model_obj: object, slice_image_shape: tuple, rect_thickness: int,
        font_size: int, font_thickness: int, hog_params: dict
) -> np.ndarray:
    for color_space, rects_list in color_rects_data.items():
        for rect in rects_list:
            sign = resize(
                image=image[rect['y']:rect['y']+rect['h'], rect['x']:rect['x']+rect['w']],
                output_shape=slice_image_shape,
                preserve_range=True
            ).astype(np.uint8)

            a = get_hog_descriptors(images=[rgb2hsv(sign)], hog_params=hog_params)
            pred = model_obj.predict(a)

            cv2.rectangle(
                image,
                (rect['x'], rect['y']),
                (rect['x'] + rect['w'], rect['y'] + rect['h']),
                COLOR[color_space],
                rect_thickness
            )

            cv2.putText(
                image,
                pred[0],
                (rect['x'] + rect['w'], rect['y']),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                COLOR[color_space],
                font_thickness,
                cv2.LINE_AA
            )


def add_rects_with_pred_on_images(
        rects_data: dict, model_obj: object, source_image_shape: tuple, home_path: str, slice_image_shape: tuple,
        rect_thickness: int, font_size: int, font_thickness: int, hog_params: dict
) -> list:
    result = []
    result_append = result.append

    for i, (filename, all_color_rects) in enumerate(rects_data.items()):
        image = resize(
            image=imread(join(home_path, filename)),
            output_shape=source_image_shape,
            preserve_range=True
        ).astype(np.uint8)

        add_rects_with_pred_on_image(
            image=image,
            color_rects_data=all_color_rects,
            model_obj=model_obj,
            slice_image_shape=slice_image_shape,
            rect_thickness=rect_thickness,
            font_size=font_size,
            font_thickness=font_thickness,
            hog_params=hog_params
        )

        result_append({'file': filename, 'img': image})

    return result
