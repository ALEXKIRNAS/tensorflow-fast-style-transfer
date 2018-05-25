import logging
import  click
import time
from typing import Callable

import cv2

from utils.demo_utils import StyleTransferDemo

logging.basicConfig(level=logging.INFO)


def camera_demo(transform_fn: Callable):
    cap = cv2.VideoCapture(0)

    while True:
        frame_start = time.time()

        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        styled_frame = transform_fn(rgb_frame)
        styled_frame = cv2.cvtColor(styled_frame, cv2.COLOR_RGB2BGR)

        frame_end = time.time()
        frame_time = frame_end - frame_start
        fps = 1. / frame_time

        logging.info('Frame time - {:.2f} ms - FPS - {:.2f}'.format(
            frame_time * 1000., fps
        ))

        cv2.imshow('CameraDemo', styled_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@click.command()
@click.option('--model',
              default='../model/rain_princess.pb')
def main(model):
    transformer = StyleTransferDemo(
        model_path=model,
        input_shape=(360, 640),
        scope='style_transfer_cnn'
    )
    camera_demo(transformer)


if __name__ == '__main__':
    main()