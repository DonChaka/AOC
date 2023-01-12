import cv2

from utils.classifier import load_dataset, get_trained_model
from utils.globals import *
from utils.contour import match_rects_to_images, add_rects_on_image
from utils.prediction import add_rects_with_pred_on_image

COLOR_SETS = True

dataset = load_dataset(path=DATASET_PATH, hog_params=HOG_PARAMS, slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE)
model = get_trained_model(
        x_train_data=dataset[0],
        y_train_data=dataset[1],
        model_class=CLASSIFIER_MODEL_CLASS,
        model_params=CLASSIFIER_PARAMS,
        model_search=CLASSIFIER_SEARCH_BEST_MODEL,
        search_params=CLASSIFIER_SEARCH_BEST_MODEL_PARAMS
    )

for video_name in ('example_day', 'example_night'):
    cap = cv2.VideoCapture(join('videos', f'{video_name}.mp4'))

    if not cap.isOpened():
        print('Failed')
        exit(1)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            rects = match_rects_to_images(image=frame)
            if len(rects):
                add_rects_with_pred_on_image(
                    image=frame,
                    rects_data=rects,
                    model=model,
                    slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE,
                    rect_thickness=SIGN_RECTANGLE_THICKNESS,
                    font_size=FONT_SIZE,
                    font_thickness=FONT_THICKNESS,
                    hog_params=HOG_PARAMS
                )

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
