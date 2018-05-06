import cv2
import numpy as np
import tensorflow as tf

from utils.tf_session_helper import TFSessionHandler
from typing import Tuple


class StyleTransferDemo(TFSessionHandler):
    """
    Style Transfer demo helper.
    """

    def __init__(self,
                 model_path: str,
                 input_shape: Tuple[int, int],
                 scope: str = ''):
        """
        :param model_path: path to model protobuf file.
        :param input_shape: model input shape (height, width).
        :param scope: a name scope of model.
        """

        super(StyleTransferDemo, self).__init__(scope=scope)
        self.model_path = model_path
        self.input_shape = input_shape
        self.model = self._load_model()
        self.scope = scope + '/' if scope else ''

        self.input_image_tensor = self.model.get_tensor_by_name(
            self.scope + 'content:0'
        )
        self.output_image_tensor = self.model.get_tensor_by_name(
            self.scope + 'styled_image:0'
        )

    def _load_model(self) -> tf.Graph:
        """
        Load model from serialized file and update session graph
        :return: graph with loaded model.
        """

        graph = self._load_protobuf(self.model_path)
        return graph

    def transform(self, frames_rgb: np.ndarray) -> np.ndarray:
        """
        Apply stylization batch of images.
        :param frames_rgb: batch of images.
        :return: styled images.
        """

        if frames_rgb.ndim == 3:
            frames_rgb = np.expand_dims(frames_rgb, axis=0)
        elif frames_rgb.ndim != 4:
            raise ValueError('Inputs must be 3 or 4 dimensional.'
                             'Get input with {} dims.'.format(frames_rgb.ndim))

        frames_rgb = [
            cv2.resize(frame,
                       dsize=(self.input_shape[1], self.input_shape[0]))
            for frame in frames_rgb
        ]

        styled_images = self.sess.run(
            fetches=self.output_image_tensor,
            feed_dict={
                self.input_image_tensor: frames_rgb
            })

        styled_images = np.clip(styled_images, a_min=0., a_max=255.)
        styled_images = styled_images.astype(dtype=np.uint8)
        styled_images = np.squeeze(styled_images)

        return styled_images

    def __call__(self, frames_rgb: np.ndarray) -> np.ndarray:
        return self.transform(frames_rgb)
