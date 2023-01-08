from os.path import join
from time import time

import cv2
from sklearn.ensemble import RandomForestClassifier

from utis.classifier import load_dataset, get_trained_model
from utis.contour import match_rect_to_contour, get_base_contours
from utis.prediction import add_rects_with_pred_on_image

OUTPUT_SOURCE_IMAGE_SHAPE = (900, 1200, 3)
SIGN_RECTANGLE_OFFSET = 5
SIGN_MIN_SIZE = 10
SIGN_MATCH_DISTANCE_EPSILON = 0.05
RECT_THICKNESS = 2
DATASET_PATH = join('Images', 'polishDataset')
HOG_PARAMS = {
    'orientation': 8,
    'pixels_per_cell': (5, 5),
    'cells_per_block': (8, 8)
}
OUTPUT_SLICE_IMAGE_SHAPE = (40, 40, 3)
CLASSIFIER_MODEL_CLASS = RandomForestClassifier
CLASSIFIER_SEARCH_BEST_MODEL = False
CLASSIFIER_PARAMS = {
    'n_estimators': 50,
    'criterion': 'entropy',
    'max_depth': 10,
    'min_samples_split': 2
}
FONT_SIZE = 1
FONT_THICKNESS = 2

base_contours = get_base_contours(join('Images', 'Shapes'))
cap = cv2.VideoCapture(join('videos', 'example_day.mp4'))

if not cap.isOpened():
    print('Failed')
    exit(1)

match_rect_times = []
match_rect_times_append = match_rect_times.append

add_rect_to_image_times = []
add_rect_to_image_times_append = add_rect_to_image_times.append


x_train, y_train, _, _ = load_dataset(
    path=DATASET_PATH,
    hog_params=HOG_PARAMS,
    slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE
)

model = get_trained_model(
    x_train_data=x_train,
    y_train_data=y_train,
    model_class=CLASSIFIER_MODEL_CLASS,
    model_params=CLASSIFIER_PARAMS,
    model_search=CLASSIFIER_SEARCH_BEST_MODEL
)

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
            model_obj=model,
            slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE,
            rect_thickness=RECT_THICKNESS,
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
