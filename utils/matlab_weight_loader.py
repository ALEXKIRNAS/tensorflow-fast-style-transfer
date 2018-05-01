import numpy as np
import scipy.io
from typing import Iterable, Tuple


class MatlabWeightsParser(object):
    """
    Class for handling matconvnet weights.
    """
    def __init__(self, weights_path: str, layer_names: Iterable[str]):
        """
        :param weights_path: path to matconvnet file.
        :param layer_names: names of convnet for which weight are loading.
        """

        self._data = scipy.io.loadmat(weights_path)['layers'][0]

        self._weights = {
            layer: MatlabWeightsParser.get_conv_layer_weights(self._data[index])
            for index, layer in enumerate(layer_names)
            if layer.startswith('conv')
        }

    def get_layer_weights(self, layer_name: str) -> np.ndarray:
        """
        Gets weights for layer by name.
        :param layer_name: layer name for which need load weights.
        :return: layer weights.
        """

        if layer_name not in self._weights:
            raise KeyError('Invalid layer name {}.'.format(
                layer_name
            ))

        return self._weights[layer_name]

    @staticmethod
    def get_conv_layer_weights(layer_weights: np.ndarray) -> Tuple[np.ndarray]:
        """
        Loads convolution weights for matconvnet format into TensorFlow format.
        :param layer_weights: matconvnet weights.
        :return: TensorFlow weights.
        """

        kernel_weights = layer_weights[0][0][2][0][0]

        # matconvnet: layer_weights are
        # [width, height, in_channels, out_channels]

        # tensorflow: layer_weights are
        # [height, width, in_channels, out_channels]

        kernel_weights = np.transpose(kernel_weights, (1, 0, 2, 3))

        bias_weights = np.array(layer_weights[0][0][2][0][1])
        bias_weights = bias_weights.reshape(-1)

        return kernel_weights, bias_weights
