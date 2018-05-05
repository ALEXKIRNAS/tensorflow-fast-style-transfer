import tensorflow as tf
from utils.layers_utils import (reflection_padding, convolution_layer,
                                instance_norm_leaky_relu, residual_block,
                                deconvolution_layer, stylization_layer)


class TransformationNet(object):
    """
    Defines transformation network.
    """
    def __init__(self, scope: str = None):
        self.scope = scope

    @staticmethod
    def preprocess_inputs(inputs: tf.Tensor) -> tf.Tensor:
        return inputs / 255.

    def build_transformation_net(self, inputs: tf.Tensor) -> tf.Tensor:

        with tf.name_scope(self.scope):
            padded_image = reflection_padding(inputs, padding_size=24)

            conv0 = convolution_layer(
                padded_image,
                num_filters=3,
                filter_size=(3, 3),
                name='Conv0'
            )
            conv0_bn = instance_norm_leaky_relu(conv0, name='Conv0_bn')

            conv1 = convolution_layer(
                conv0_bn,
                num_filters=16,
                filter_size=(7, 7),
                stride=(1, 1),
                name='Conv1'
            )
            conv1_bn = instance_norm_leaky_relu(conv1, name='Conv1_bn')

            conv2 = convolution_layer(
                conv1_bn,
                num_filters=32,
                filter_size=(3, 3),
                stride=(2, 2),
                name='Conv2'
            )
            conv2_bn = instance_norm_leaky_relu(conv2, name='Conv2_bn')

            conv3 = convolution_layer(
                conv2_bn,
                num_filters=64,
                filter_size=(3, 3),
                stride=(2, 2),
                name='Conv3'
            )

            residual_1 = residual_block(
                conv3,
                num_filters=64,
                name='Residual_1'
            )

            residual_2 = residual_block(
                residual_1,
                num_filters=64,
                name='Residual_2'
            )

            residual_3 = residual_block(
                residual_2,
                num_filters=64,
                name='Residual_3'
            )
            residual_3_bn = instance_norm_leaky_relu(
                residual_3,
                name='Residual_3_bn'
            )

            deconv_1 = deconvolution_layer(
                residual_3_bn,
                num_filters=32,
                filter_size=(3, 3),
                stride=(2, 2),
                name='Deconv_1')
            deconv_1 = instance_norm_leaky_relu(deconv_1, name='Deconv_1_bn')

            deconv_2 = deconvolution_layer(
                deconv_1,
                num_filters=16,
                filter_size=(3, 3),
                stride=(2, 2),
                name='Deconv_2'
            )
            deconv_2 = instance_norm_leaky_relu(deconv_2, name='Deconv_2_bn')

            styled_image = stylization_layer(
                deconv_2,
                num_filters=3,
                filter_size=(5, 5),
                name='Stylization'
            )
            styled_image = instance_norm_leaky_relu(
                styled_image,
                activation=False,
                name='Stylization_bn'
            )

            styled_image = tf.nn.tanh(styled_image) + 1
            styled_image = tf.multiply(
                x=styled_image,
                y=(255. / 2),
                name='styled_image'
            )

        return styled_image
