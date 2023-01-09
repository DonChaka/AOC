from time import time

import cv2

from utils.globals import *
from utils.classifier import load_dataset, get_trained_model
from utils.contour import match_rect_to_contour, get_base_contours
from utils.prediction import add_rects_with_pred_on_image


COLOR_SETS = True

cap = cv2.VideoCapture(join('videos', 'example_day.mp4'))

if not cap.isOpened():
    print('Failed')
    exit(1)


train_test_sets = {
    'red': load_dataset(
        path=COLORED_DATASET_PATHS['red'],
        hog_params=HOG_PARAMS,
        slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE
    ),
    'blue': load_dataset(
        path=COLORED_DATASET_PATHS['blue'],
        hog_params=HOG_PARAMS,
        slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE
    ),
    'yellow': load_dataset(
        path=COLORED_DATASET_PATHS['yellow'],
        hog_params=HOG_PARAMS,
        slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE
    )
}

models = {}

for color_space in COLORED_DATASET_PATHS:
    models[color_space] = get_trained_model(
        x_train_data=train_test_sets[color_space][0],
        y_train_data=train_test_sets[color_space][1],
        model_class=CLASSIFIER_MODEL_CLASS,
        model_params=CLASSIFIER_PARAMS,
        model_search=CLASSIFIER_SEARCH_BEST_MODEL,
        search_params=CLASSIFIER_SEARCH_BEST_MODEL_PARAMS
    )

base_contours = get_base_contours(join('Images', 'Shapes'))
match_rect_times = []
match_rect_times_append = match_rect_times.append

add_rect_to_image_times = []
add_rect_to_image_times_append = add_rect_to_image_times.append

while cap.isOpened():
    ret, frame = cap.read()
    source_shape = frame.shape

    if ret:
        start = time()
        rects = match_rect_to_contour(
            image=frame,
            source_image_shape=OUTPUT_SOURCE_IMAGE_SHAPE,
            base_contours=base_contours,
            rect_offset=SIGN_RECTANGLE_OFFSET,
            rect_min_size=SIGN_MIN_SIZE,
            sign_min_match_distance=SIGN_MATCH_DISTANCE_EPSILON
        )
        match_rect_times_append(time() - start)

        start = time()
        add_rects_with_pred_on_image(
            image=frame,
            color_rects_data=rects,
            models_dict=models,
            slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE,
            rect_thickness=SIGN_RECTANGLE_THICKNESS,
            font_size=FONT_SIZE,
            font_thickness=FONT_THICKNESS,
            hog_params=HOG_PARAMS
        )
        add_rect_to_image_times_append(time() - start)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()

print(f'Avg match time: {sum(match_rect_times) / len(match_rect_times)}s')
print(f'Avg draw: {sum(add_rect_to_image_times) / len(add_rect_to_image_times)}s')
