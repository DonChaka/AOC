from os import listdir
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.transform import resize


def __histogram_equalization(data: np.ndarray) -> np.ndarray:
    if len(data.shape) > 2 and data.shape[2] > 1:
        red, green, blue = cv2.split(data)

        out_red = cv2.equalizeHist(red)
        out_green = cv2.equalizeHist(green)
        out_blue = cv2.equalizeHist(blue)

        return cv2.merge((out_red, out_green, out_blue))

    return cv2.equalizeHist(data)


def __get_red_mask(data: np.ndarray) -> np.ndarray:
    red_low_boundary = np.array([310 / 360, 55 / 100, 55 / 100])
    red_high_boundary = np.array([360 / 360, 100 / 100, 100 / 100])

    red_mask = cv2.inRange(data, red_low_boundary, red_high_boundary)

    red_low_boundary = np.array([0 / 360, 50 / 100, 55 / 100])
    red_high_boundary = np.array([15 / 360, 100 / 100, 100 / 100])

    return cv2.inRange(data, red_low_boundary, red_high_boundary) | red_mask


def __get_blue_mask(data: np.ndarray) -> np.ndarray:
    blue_low_boundary = np.array([195 / 360, 40 / 100, 55 / 100])
    blue_high_boundary = np.array([220 / 360, 100 / 100, 100 / 100])

    return cv2.inRange(data, blue_low_boundary, blue_high_boundary)


def __get_yellow_mask(data: np.ndarray) -> np.ndarray:
    yellow_low_boundary = np.array([35 / 360, 60 / 100, 60 / 100])
    yellow_high_boundary = np.array([65 / 360, 100 / 100, 100 / 100])

    return cv2.inRange(data, yellow_low_boundary, yellow_high_boundary)


def __get_mask_per_color_space() -> dict:
    return {
        'red': __get_red_mask,
        'yellow': __get_yellow_mask,
        'blue': __get_blue_mask
    }


def __get_mask(img: np.ndarray, rule: str) -> np.ndarray:
    return __get_mask_per_color_space().get(rule)(data=img)


def __display_images_masks(img: np.ndarray, file_name: str) -> None:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2, figsize=(20, 15))

    ax0.imshow(img)
    ax0.set_title(file_name)
    ax0.axis('off')

    hsv_image = rgb2hsv(__histogram_equalization(data=img))

    ax1.imshow(__get_mask(img=hsv_image, rule='red'), cmap=plt.cm.gray)
    ax1.set_title('red channel')
    ax1.axis('off')

    ax2.imshow(__get_mask(img=hsv_image, rule='blue'), cmap=plt.cm.gray)
    ax2.set_title('blue channel')
    ax2.axis('off')

    ax3.imshow(__get_mask(img=hsv_image, rule='yellow'), cmap=plt.cm.gray)
    ax3.set_title('yellow channel')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def __display_merged_masks(img: np.ndarray, file_name: str) -> None:
    fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 15))

    ax0.imshow(img)
    ax0.set_title(file_name)
    ax0.axis('off')

    hsv_image = rgb2hsv(__histogram_equalization(data=img))
    mask = __get_mask(img=hsv_image, rule='red') | __get_mask(img=hsv_image, rule='blue') | \
           __get_mask(img=hsv_image, rule='yellow')

    ax1.imshow(mask, cmap=plt.cm.gray)
    ax1.set_title('merged mask')
    ax1.axis('off')

    plt.tight_layout()
    plt.show()


def plot_all_mask_per_image(home_path: str, source_image_shape: tuple) -> None:
    for file in listdir(home_path):
        __display_images_masks(
            img=resize(
                image=imread(join(home_path, file)),
                output_shape=source_image_shape,
                preserve_range=True
            ).astype(np.uint8),
            file_name=file
        )


def plot_merged_mask_per_image(home_path: str, source_image_shape: tuple) -> None:
    for file in listdir(home_path):
        __display_merged_masks(
            img=resize(
                image=imread(join(home_path, file)),
                output_shape=source_image_shape,
                preserve_range=True
            ).astype(np.uint8),
            file_name=file
        )


def get_mask_dict(img: np.ndarray) -> dict:
    return {
        'red': __get_mask(img=img, rule='red'),
        'blue': __get_mask(img=img, rule='blue'),
        'yellow': __get_mask(img=img, rule='yellow')
    }
