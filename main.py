import cv2

from utils.globals import *
from utils.contour import add_rects_on_image, match_rects_to_images

COLOR_SETS = True

for video_name in ('example_day', 'example_night'):
    cap = cv2.VideoCapture(join('videos', f'{video_name}.mp4'))

    if not cap.isOpened():
        print('Failed')
        exit(1)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            rects = match_rects_to_images(image=frame)
            add_rects_on_image(img=frame, all_rects=rects)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
