import cv2

from utils.globals import *
from utils.contour import add_rects_on_image, match_rects_to_images

COLOR_SETS = True
FRAMES_BETWEEN_SAVE = 60
SAVE_MODE = 0  # 0 - clear, 1 - rects, 2 - rects with pred

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
                add_rects_on_image(img=frame, all_rects=rects)
            elif SAVE_MODE == 2:
                raise NotImplementedError

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
