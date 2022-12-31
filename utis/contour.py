from collections import defaultdict
from itertools import product
from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize

from utis.colours import COLOR
from utis.mask import get_mask_dict_for_img


def __is_contour_match(distance: float, min_match_distance: float) -> bool:
    return distance < min_match_distance


def __get_rect_dims(contour: np.ndarray, x_offset: int = 0, y_offset: int = 0) -> tuple:
    x, y, w, h = cv2.boundingRect(contour)
    return x - x_offset, y - y_offset, w + x_offset, h + y_offset


def __match_rect_to_contour(
        home: str, file_name: str, source_image_shape: tuple, base_contours: dict, rect_offset: int, rect_min_size: int,
        sign_min_match_distance: float
) -> dict:
    result = defaultdict(list)

    img_masks = get_mask_dict_for_img(
        img=resize(
            image=imread(join(home, file_name)),
            output_shape=source_image_shape,
            preserve_range=True
        ).astype(np.uint8)
    )

    for color_space, mask in img_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for base, contour in product(base_contours.values(), contours):
            distance = cv2.matchShapes(base, contour, cv2.CONTOURS_MATCH_I3, 0)

            if __is_contour_match(distance=distance, min_match_distance=sign_min_match_distance):
                x_pos, y_pos, width, height = __get_rect_dims(
                    contour=contour,
                    x_offset=rect_offset,
                    y_offset=rect_offset
                )

                if width > rect_min_size and height > rect_min_size:
                    result[color_space].extend([{'x': x_pos, 'y': y_pos, 'w': width, 'h': height}])

    return dict(result)


def __add_rects_on_image(image: np.ndarray, color_rects_data: dict, rect_thickness: float) -> None:
    for color_space, rects_list in color_rects_data.items():
        for rect in rects_list:
            cv2.rectangle(
                image,
                (rect['x'], rect['y']),
                (rect['x'] + rect['w'], rect['y'] + rect['h']),
                COLOR[color_space],
                rect_thickness
            )


def get_base_contours(path: str) -> dict:
    result = {}

    for filename in listdir(path):
        image_path = join(path, filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 150, 255, 0)
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result[f'{filename.removesuffix(".png")}'] = contours[0]

    return result


def get_rects_for_images(
        home: str, source_image_shape: tuple, base_contours: dict, rect_offset: int, rect_min_size: int,
        sign_min_match_distance: float
) -> dict:
    image_rects = {}

    for filename in listdir(home):
        image_rects[filename] = __match_rect_to_contour(
            home=home,
            file_name=filename,
            source_image_shape=source_image_shape,
            base_contours=base_contours,
            rect_offset=rect_offset,
            rect_min_size=rect_min_size,
            sign_min_match_distance=sign_min_match_distance
        )

    return image_rects


def add_rects_on_images(rects_data: dict, home_path: str, source_image_shape: tuple, rect_thickness: float) -> list:
    result = []
    result_append = result.append

    for i, (filename, all_color_rects) in enumerate(rects_data.items()):
        image = resize(
            image=imread(join(home_path, filename)),
            output_shape=source_image_shape,
            preserve_range=True
        ).astype(np.uint8)

        __add_rects_on_image(
            image=image,
            color_rects_data=all_color_rects,
            rect_thickness=rect_thickness
        )

        result_append({'file': filename, 'img': image})

    return result


def plot_images(images: list) -> None:
    fig, axes = plt.subplots(ncols=2, nrows=len(images) // 2 + 1, figsize=(24, 25))
    axes = axes.ravel()

    for i, image_data in enumerate(images):
        axes[i].imshow(image_data['img'], cmap=plt.cm.brg)
        axes[i].set_title(image_data['file'])
        axes[i].axis('off')

    for i in range(len(images), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
