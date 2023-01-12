import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.feature import hog
from skimage.transform import resize


def __get_hog_of_image(img: np.ndarray, orientation: int, pixels_per_cell: tuple, cells_per_block: tuple) -> np.ndarray:
    return hog(
        image=img,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=-1
    )


def display_hogs_of_rects(images_with_rects_data: tuple, slice_image_shape: str, hog_params: dict) -> None:
    for rect_data in images_with_rects_data:
        plt.imshow(rect_data['img'])
        plt.title(rect_data['file_name'])

        for rect in rect_data['rects']:
            sign = resize(
                image=rect_data['img'][rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]],
                output_shape=slice_image_shape,
                preserve_range=True
            ).astype(np.uint8)

            fd, hog_image = __get_hog_of_image(sign, **dict(hog_params.items()))
            # Rescale histogram for better display
            hog_image_rescaled = rescale_intensity(hog_image, in_range=(0, 10))

            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

            ax1.axis('off')
            ax1.imshow(sign)
            ax1.set_title(f'Slice input image of {rect_data["file_name"]}')

            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
