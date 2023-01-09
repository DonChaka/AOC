from os import listdir
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.transform import resize


def __p2f(value: int) -> float:
    return value / 100


def __make_image_binary(data: np.ndarray) -> np.ndarray:
    data = np.sum(data, axis=2, keepdims=False, dtype=np.uint8)
    data[data > 0] = 1
    return data


def __get_image_mask(data: np.ndarray, idxes: np.ndarray) -> np.ndarray:
    result = np.zeros_like(data)
    result[idxes] = data[idxes]
    return result


def __get_red_mask_rules(data: np.ndarray) -> tuple:
    hue = ((0 <= data[:, :, 0]) & (data[:, :, 0] <= 10 / 360)) | ((300 / 360 <= data[:, :, 0]) & (data[:, :, 0] <= 1))
    sat = (__p2f(25) <= data[:, :, 1]) & (data[:, :, 1] <= __p2f(250))
    value = (__p2f(30) < data[:, :, 2]) & (data[:, :, 2] <= __p2f(200))
    return hue, sat, value


def __get_blue_mask_rules(data: np.ndarray) -> tuple:
    hue = (190 / 360 <= data[:, :, 0]) & (data[:, :, 0] <= 260 / 360)
    sat = (__p2f(20) <= data[:, :, 1]) & (data[:, :, 1] <= __p2f(250))
    value = (__p2f(35) < data[:, :, 2]) & (data[:, :, 2] <= __p2f(128))
    return hue, sat, value


def __get_yellow_mask_rules(data: np.ndarray) -> tuple:
    hue = (40 / 360 <= data[:, :, 0]) & (data[:, :, 0] <= 65 / 360)
    sat = (__p2f(60) <= data[:, :, 1]) & (data[:, :, 1] <= __p2f(250))
    value = (__p2f(60) < data[:, :, 2]) & (data[:, :, 2] <= __p2f(128))
    return hue, sat, value


def __get_mask(img: np.ndarray, rule: str) -> np.ndarray:
    hue, sat, value = __get_mask_rules().get(rule)(data=rgb2hsv(img))
    return __make_image_binary(
        data=__get_image_mask(data=img, idxes=hue & sat & value)
    )


def __display_images_masks(image: np.ndarray, filename: str) -> None:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2, figsize=(20, 15))

    ax0.imshow(image)
    ax0.set_title(filename)
    ax0.axis('off')

    ax1.imshow(__get_mask(img=image, rule='red'), cmap=plt.cm.gray)
    ax1.set_title('red channel')
    ax1.axis('off')

    ax2.imshow(__get_mask(img=image, rule='blue'), cmap=plt.cm.gray)
    ax2.set_title('blue channel')
    ax2.axis('off')

    ax3.imshow(__get_mask(img=image, rule='yellow'), cmap=plt.cm.gray)
    ax3.set_title('yellow channel')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def __display_merged_masks(image: np.ndarray, filename: str) -> None:
    fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 15))

    ax0.imshow(image)
    ax0.set_title(filename)
    ax0.axis('off')

    mask = __get_mask(img=image, rule='red') | __get_mask(img=image, rule='blue') | __get_mask(img=image, rule='yellow')

    ax1.imshow(mask, cmap=plt.cm.gray)
    ax1.set_title('merged mask')
    ax1.axis('off')

    plt.tight_layout()
    plt.show()


def __get_mask_rules() -> dict:
    return {
        'red': __get_red_mask_rules,
        'yellow': __get_yellow_mask_rules,
        'blue': __get_blue_mask_rules
    }


def plot_all_mask_per_image(home_path: str, source_image_shape: tuple) -> None:
    for file in listdir(home_path):
        __display_images_masks(
            image=resize(
                image=imread(join(home_path, file)),
                output_shape=source_image_shape,
                preserve_range=True
            ).astype(np.uint8),
            filename=file
        )


def plot_merged_mask_per_image(home_path: str, source_image_shape: tuple) -> None:
    for file in listdir(home_path):
        __display_merged_masks(
            image=resize(
                image=imread(join(home_path, file)),
                output_shape=source_image_shape,
                preserve_range=True
            ).astype(np.uint8),
            filename=file
        )


def get_mask_dict_for_img(img: np.ndarray) -> dict:
    return {
        'red': __get_mask(img=img, rule='red').astype(np.uint8),
        'yellow': __get_mask(img=img, rule='yellow').astype(np.uint8),
        'blue': __get_mask(img=img, rule='blue').astype(np.uint8)
    }
