from os import listdir, cpu_count
from os.path import join
from time import time
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support, \
    classification_report
from sklearn.model_selection import GridSearchCV


__CPUS = cpu_count()
pd.set_option('display.max_rows', None)


def __read_images_from_path(path: str, slice_image_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    x_append, y_append = x.append, y.append

    for dir_name in listdir(path):
        for f_name in listdir(join(path, dir_name)):
            image = rgb2hsv(resize(
                image=imread(fname=join(path, dir_name, f_name)),
                output_shape=slice_image_shape,
                preserve_range=True
            ).astype(np.uint8))

            x_append(image)
            y_append(dir_name)

    return np.array(x), np.array(y)


def __get_hog_of_image(
        image: np.ndarray, orientation: int, pixels_per_cell: tuple, cells_per_block: tuple
) -> np.ndarray:
    return hog(
        image=image,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block
    )


def __get_model_basic_config(model_class: callable) -> dict:
    if isinstance(model_class(), RandomForestClassifier):
        return {'n_jobs': __CPUS // 2}

    return {}


def __calculate_sensitivity_and_specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    results = []
    classes = np.unique(y_true)

    for class_id in classes:
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true=y_true == class_id,
            y_pred=y_pred == class_id,
            pos_label=True,
            average=None
        )

        results.append([class_id, recall[0], recall[1]])

    return pd.DataFrame(results, columns=['class', 'sensitivity', 'specificity'])


def __get_basic_metrics(data: pd.DataFrame, test_data: np.ndarray, pred_data: np.ndarray) -> pd.DataFrame:
    return data[-3:].drop(columns=['support']).T.assign(
        temp=balanced_accuracy_score(y_true=test_data, y_pred=pred_data, adjusted=True)
    ).rename(columns={'temp': 'balanced adjusted accuracy'}).T


def get_hog_descriptors(images: Iterable, hog_params: dict) -> np.ndarray:
    descriptors = []
    descriptors_append = descriptors.append

    for image in images:
        descriptors_append(np.concatenate((
            __get_hog_of_image(image=image[:, :, 0], **dict(hog_params.items())),
            __get_hog_of_image(image=image[:, :, 1], **dict(hog_params.items())),
            __get_hog_of_image(image=image[:, :, 2], **dict(hog_params.items()))
        )))

    return np.array(descriptors)


def load_dataset(
        path: str, hog_params: dict, slice_image_shape: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_data, y_train_data = __read_images_from_path(path=join(path, 'train'), slice_image_shape=slice_image_shape)
    x_test_data, y_test_data = __read_images_from_path(path=join(path, 'test'), slice_image_shape=slice_image_shape)

    return get_hog_descriptors(images=x_train_data, hog_params=hog_params), y_train_data,\
        get_hog_descriptors(images=x_test_data, hog_params=hog_params), y_test_data


def get_trained_model(
        x_train_data: np.ndarray, y_train_data: np.ndarray, model_class: callable, model_params: Optional[dict] = None,
        model_search: bool = False, search_params: Optional[dict] = None
) -> object:
    if model_search:
        if search_params is None:
            raise ValueError('Parameter search_params can not be None')

        grid_search = GridSearchCV(
            estimator=model_class(**dict(__get_model_basic_config(model_class).items())),
            param_grid=search_params,
            verbose=3
        )

        start = time()
        grid_search.fit(x_train_data, y_train_data)
        print(f'Search took {round(time() - start, 2)}s.\nBest params: {grid_search.best_params_}')
        return grid_search.best_estimator_
    else:
        model_obj = model_class(**dict(__get_model_basic_config(model_class).items()), **dict(model_params.items()))
        _ = model_obj.fit(x_train_data, y_train_data)
        return model_obj


def evaluate_model(model: object, x_test: np.ndarray, y_test: np.ndarray, show_details: bool = False) -> None:
    y_pred = model.predict(x_test)

    result_df = pd.DataFrame(classification_report(y_true=y_test, y_pred=y_pred, output_dict=True, zero_division=1)).T

    basic_metrics = __get_basic_metrics(data=result_df, test_data=y_test, pred_data=y_pred)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    display(basic_metrics)
    sns.heatmap(cm)
    plt.show()

    if show_details:
        temp_df = __calculate_sensitivity_and_specificity_per_class(y_true=y_test, y_pred=y_pred)
        result_df = result_df.T.drop(columns=['accuracy', 'macro avg', 'weighted avg']).T

        result_df = result_df.assign(
            sensitivity=temp_df['sensitivity'].tolist(),
            specificity=temp_df['specificity'].tolist()
        )
        display(result_df)
