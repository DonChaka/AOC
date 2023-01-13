import cv2

from utils.classifier import load_dataset, get_trained_model
from utils.globals import *
from utils.contour import add_rects_on_image, match_rects_to_images
from utils.prediction import add_rects_with_pred_on_image

COLOR_SETS = True
FRAMES_BETWEEN_SAVE = 60
SAVE_MODE = 0  # 0 - clear, 1 - rects, 2 - rects with pred

if SAVE_MODE == 2:
    dataset = load_dataset(path=DATASET_PATH, hog_params=HOG_PARAMS, slice_image_shape=OUTPUT_SLICE_IMAGE_SHAPE)
    model = get_trained_model(
            x_train_data=dataset[0],
            y_train_data=dataset[1],
            model_class=CLASSIFIER_MODEL_CLASS,
            model_params=CLASSIFIER_PARAMS,
            model_search=CLASSIFIER_SEARCH_BEST_MODEL,
            search_params=CLASSIFIER_SEARCH_BEST_MODEL_PARAMS
        )

file_name = 0
for video_name in ('example_day', 'example_night'):
    cap = cv2.VideoCapture(join('videos', f'{video_name}.mp4'))

    if not cap.isOpened():
        print('Failed')
        exit(1)

    frame_counter = 0
    path = join('Images', 'vid_frames', video_name)

    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter += 1

        if ret:
            if SAVE_MODE == 0:
                pass
            elif SAVE_MODE == 1:
                rects = match_rects_to_images(image=frame)
                if len(rects):
                    add_rects_on_image(img=frame, all_rects=rects)
            elif SAVE_MODE == 2:
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
