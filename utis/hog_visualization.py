from skimage import exposure
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from os.path import join


def __get_hog_of_image(
        image: np.ndarray, orientation: int, pixels_per_cell: tuple, cells_per_block: tuple
) -> np.ndarray:
    return hog(
        image=image,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=-1
    )


def display_hogs_of_rects(
        rects: dict, source_image_shape: tuple, home_path: str, slice_image_shape: str, hog_params: dict
):
    for filename in rects:
        image = resize(
            image=imread(join(home_path, filename)),
            output_shape=source_image_shape,
            preserve_range=True
        ).astype(np.uint8)

        plt.imshow(image)
        plt.show()

        for color_base in rects[filename]:
            for rect in rects[filename][color_base]:
                sign = resize(
                    image=image[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']],
                    output_shape=slice_image_shape,
                    preserve_range=True
                ).astype(np.uint8)

                fd, hog_image = __get_hog_of_image(sign, **dict(hog_params.items()))

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

                ax1.axis('off')
                ax1.imshow(sign)
                ax1.set_title('Input image')

                # Rescale histogram for better display
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

                ax2.axis('off')
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                plt.show()
