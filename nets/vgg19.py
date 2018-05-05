from typing import Dict

import numpy as np
import tensorflow as tf

from utils.matlab_weight_loader import MatlabWeightsParser


class VGG19(object):
    """
    Define VGG19 model.
    """

    LAYERS_NAME = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    IMG_MEAN = np.array([123.68, 116.779, 103.939])

    def __init__(self, weights_path: str):
        self._data = MatlabWeightsParser(
            weights_path=weights_path,
            layer_names=VGG19.LAYERS_NAME
        )

    def preprocess_inputs(self, image: tf.Tensor) -> tf.Tensor:
        """
        Preprocess input image for model inference.
        """
        return image - self.IMG_MEAN

    def build(self,
              input_tensor: tf.Tensor,
              scope: str = None) -> Dict[str, tf.Tensor]:
        """
        Builds model.
        :param input_tensor: network inputs.
        :param scope: network name scope.
        :return: dict with network endpoints.
        """
        endpoints = {}
        net = input_tensor

        with tf.variable_scope(scope):
            for name in self.LAYERS_NAME:
                kind = name[:4]

                if kind == 'conv':
                    kernel_weights, bias_weights = self._data.get_layer_weights(
                        layer_name=name
                    )

                    net = tf.nn.conv2d(
                        input=net,
                        filter=tf.constant(kernel_weights),
                        strides=(1, 1, 1, 1),
                        padding='SAME'
                    )

                    net = tf.nn.bias_add(value=net,
                                         bias=bias_weights)

                elif kind == 'relu':
                    net = tf.nn.relu(net)

                elif kind == 'pool':
                    net = tf.nn.max_pool(
                        value=net,
                        ksize=(1, 2, 2, 1),
                        strides=(1, 2, 2, 1),
                        padding='SAME')

                endpoints[name] = net

        return endpoints
