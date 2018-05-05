from typing import Tuple

import tensorflow as tf


def reflection_padding(net: tf.Tensor, padding_size: int) -> tf.Tensor:
    """
    Create reflation padding.
    :param net: layer inputs.
    :param padding_size: padding size.
    :return: padded inputs.
    """

    padding_shape = [
        [0, 0],
        [padding_size, padding_size],
        [padding_size, padding_size],
        [0, 0]
    ]
    net = tf.pad(net, paddings=padding_shape, mode="REFLECT")

    return net


def instance_normalization(net, epsilon=1e-7, name=None):
    """
    Apply instance norm.
    :param net: layer inputs.
    :param epsilon: variance epsilon for preventing zero division.
    :param name: scope name.
    :return: normalized inputs.
    """

    _, _, _, channels = net.get_shape().as_list()

    mean, variance = tf.nn.moments(net, [1,2], keep_dims=True)
    with tf.variable_scope(name):
        offset = tf.get_variable(
            name='Offset',
            initializer=tf.zeros(shape=(channels, )),
            dtype=tf.float32
        )
        scale = tf.get_variable(
            name='Scale',
            initializer=tf.ones(shape=(channels,)),
            dtype=tf.float32
        )

        normalized = tf.nn.batch_normalization(
            x=net,
            mean=mean,
            variance=variance,
            offset=offset,
            scale=scale,
            variance_epsilon=epsilon
        )

    return normalized


def instance_norm_leaky_relu(net: tf.Tensor,
                             activation: bool = True,
                             name: str = None):
    """
    Apply instance norm and Leaky-ReLu non-linearity.
    :param net: layer inputs.
    :param activation: bool flag that specify is need apply  non-linearity.
    :param name: scope name.
    :return: normalized and activated inputs.
    """

    with tf.name_scope(name=name):
        net = instance_normalization(net)

        if activation:
            net = tf.nn.leaky_relu(net, alpha=0.4)

    return net


def deconvolution_layer(net: tf.Tensor,
                        num_filters: int,
                        filter_size: Tuple[int, int],
                        stride: Tuple[int, int],
                        padding: str = 'SAME',
                        name: str = None):
    """
    Deconvolution.
    :param net: layer inputs.
    :param num_filters: number of filters.
    :param filter_size: filter size.
    :param stride: filter strides.
    :param padding: input padding.
    :param name: scope name.
    :return: deconvolved inputs.
    """

    with tf.name_scope(name=name):
        batch_size, rows, cols, in_channels = net.get_shape().as_list()
        output_rows, output_columns = int(rows * stride), int(cols * stride)

        output_shape = [batch_size, output_rows, output_columns, num_filters]
        output_shape = tf.stack(output_shape)

        if isinstance(stride, int):
            stride = [stride, stride]

        strides_shape = [1, stride[0], stride[1], 1]

        if isinstance(filter_size, int):
            filter_size = [filter_size, filter_size]

        weights_shape = [filter_size[0], filter_size[1], num_filters,
                         in_channels]

        weights = tf.get_variable(
            name='weights',
            shape=weights_shape,
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            dtype=tf.float32
        )

        net = tf.nn.conv2d_transpose(
            value=net,
            filter=weights,
            output_shape=output_shape,
            strides=strides_shape,
            padding=padding
        )

    return net


def convolution_layer(net: tf.Tensor,
                      num_filters: int,
                      filter_size: Tuple[int, int],
                      stride: Tuple[int, int] = 1,
                      padding: str = 'SAME',
                      name: str = None):
    """
    Convolution.
    :param net: layer inputs.
    :param num_filters: number of filters.
    :param filter_size: filter size.
    :param stride: filter strides.
    :param padding: input padding.
    :param name: scope name.
    :return: convolved inputs.
    """
    with tf.name_scope(name=name):
        batch_size, rows, cols, in_channels = net.get_shape().as_list()

        if isinstance(stride, int):
            stride = [stride, stride]

        strides_shape = [1, stride[0], stride[1], 1]

        if isinstance(filter_size, int):
            filter_size = [filter_size, filter_size]

        weights_shape = [filter_size[0], filter_size[1], in_channels,
                         num_filters]

        weights = tf.get_variable(
            name='weights',
            shape=weights_shape,
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            dtype=tf.float32
        )

        net = tf.nn.conv2d(
            input=net,
            filter=weights,
            strides=strides_shape,
            padding=padding
        )

    return net


def residual_block(net: tf.Tensor,
                   num_filters: int,
                   name=None):
    """
    Residual block.
    :param net: layer inputs.
    :param num_filters: number of filters.
    :param name: scope name.
    :return: inputs with applied residual.
    """

    batch, rows, cols, channels = net.get_shape().as_list()

    with tf.name_scope(name=name):
        residual = instance_norm_leaky_relu(net, name='instance_norm_1')
        residual = convolution_layer(
            residual,
            num_filters=num_filters,
            filter_size=(3, 3),
            padding='VALID',
            name='conv_1'
        )

        residual = instance_norm_leaky_relu(residual, name='instance_norm_2')
        residual = convolution_layer(
            residual,
            num_filters=num_filters,
            filter_size=(3, 3),
            padding='VALID',
            name='conv_2'
        )

        net = tf.slice(
            net,
            begin=[0, 2, 2, 0],
            size=[batch, rows - 4, cols - 4, channels]
        )

        net += residual

    return net


def stylization_layer(net: tf.Tensor,
                      num_filters: int,
                      filter_size: Tuple[int, int],
                      name: str=None):
    """
    Stylization block.
    :param net: layer inputs.
    :param num_filters: number of filters.
    :param filter_size: filter size.
    :param name: scope name.
    :return: inputs styled.
    """

    _, _, _, c = net.get_shape().as_list()

    with tf.name_scope(name):

        with tf.name_scope('left_branch'):
            left_branch = convolution_layer(
                net,
                num_filters=num_filters,
                filter_size=(1, filter_size[1]),
            )
            left_branch = instance_norm_leaky_relu(left_branch)

        with tf.name_scope('right_branch'):
            right_branch = convolution_layer(
                net,
                num_filters=num_filters,
                filter_size=(filter_size[0], 1)
            )
            right_branch = instance_norm_leaky_relu(right_branch)

        net = tf.concat([left_branch, right_branch], axis=-1)

        out_filter_size = (9 - (filter_size[0] - 1),
                           9 - (filter_size[1] - 1))
        net = convolution_layer(
            net,
            num_filters=num_filters,
            filter_size=out_filter_size)

    return net
