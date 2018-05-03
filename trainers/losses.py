import tensorflow as tf


def compute_gram_matrix(inputs: tf.Tensor, scope: str = None) -> tf.Tensor:
    """
    Compute gram matix for given tensor.
    :param inputs: input tensor.
    :param scope: name of operation scope.
    :return: gram matrix for give input.
    """

    batch_size, height, width, channels = [item.value for item in
                                           inputs.get_shape()]

    with tf.name_scope(scope):
        features = tf.reshape(
            inputs,
            shape=(batch_size, height * width, channels)
        )
        transpose_features = tf.transpose(features, perm=[0, 2, 1])
        features_cross_correlation = tf.matmul(transpose_features, features)
        features_cross_correlation /= (channels * height * width)

    return features_cross_correlation


def compute_style_loss(style_image_features: tf.Tensor,
                       generate_image_features: tf.Tensor,
                       scope: str = None) -> tf.Tensor:
    """
    Computes style loss for given inputs.
    :param style_image_features: extracted features from style image.
    :param generate_image_features extracted features from generated image.
    :param scope: name of loss scope.
    :return: computed style loss for give inputs.
    """

    batch_size, _, _, channels = [item.value for item in
                                  style_image_features.get_shape()]

    with tf.name_scope(scope):

        style_image_gram_matrix = compute_gram_matrix(
            inputs=style_image_features,
            scope='style_image_gram_matrix'
        )

        generate_image_gram_matrix = compute_gram_matrix(
            inputs=generate_image_features,
            scope='generated_image_gram_matrix'
        )

        style_loss = 2 * tf.nn.l2_loss(
            generate_image_gram_matrix - style_image_gram_matrix
        )

        style_loss /= (batch_size * channels * channels)

    return style_loss


def compute_content_loss(content_image_features: tf.Tensor,
                         generate_image_features: tf.Tensor,
                         scope: str = None) -> tf.Tensor:
    """
    Computes content loss for given inputs.
    :param content_image_features: extracted features from content image.
    :param generate_image_features extracted features from generated image.
    :param scope: name of loss scope.
    :return: computed content loss for give inputs.
    """

    batch_size, height, width, channels = [item.value for item in
                                           content_image_features.get_shape()]

    with tf.name_scope(scope):

        content_loss = 2 * tf.nn.l2_loss(
            generate_image_features - content_image_features
        )

        content_loss /= (batch_size * height * width * channels)

    return content_loss


def compute_total_variation_loss(generated_image: tf.Tensor,
                                 scope: str = None) -> tf.Tensor:
    """
    Computes total variance of given image.
    :param generated_image: input image.
    :param scope: name of loss scope.
    """

    batch_size, height, width, channels = [item.value for item in
                                           generated_image.get_shape()]

    with tf.name_scope(scope):
        vertical_total_variance = tf.nn.l2_loss(
            generated_image[:, 1:, ...] - generated_image[:, :(height - 1), ...]
        )

        horizontal_total_variance = tf.nn.l2_loss(
            generated_image[..., 1:, :] - generated_image[..., :(width - 1), :]
        )

        total_variance_loss = 2. * (
            horizontal_total_variance / ((height - 1) * width * channels) +
            vertical_total_variance / (height * (width - 1) * channels)
        )

        total_variance_loss /= batch_size
        total_variance_loss = tf.cast(total_variance_loss, tf.float32)

    return total_variance_loss
