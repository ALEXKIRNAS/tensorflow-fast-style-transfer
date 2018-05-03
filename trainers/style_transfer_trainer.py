import collections
import time
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

import transform
import utils.data_utils as utils
from trainers.losses import (compute_style_loss, compute_content_loss,
                             compute_total_variation_loss)
from vgg19 import VGG19


tf.logging.set_verbosity(tf.logging.INFO)


class StyleTransferTrainer(object):
    """
    Defines class for training neural style transformation network.
    """
    def __init__(self,
                 input_shape: Tuple[int, int],
                 content_layer_dict: Dict[str],
                 style_layer_dict: Dict[str],
                 training_images: List[str],
                 style_image: np.ndarray,
                 session: tf.Session,
                 discriminator: VGG19,
                 num_training_epochs: int,
                 batch_size: int,
                 content_loss_weight: float,
                 style_loss_weight: float,
                 total_variance_loss_weight: float,
                 learning_rate: float,
                 model_save_path: str,
                 saving_period: int):
        """
        :param input_shape: specify image input shape.
        :param content_layer_dict: dict of content layers names to their
            weights.
        :param style_layer_dict: dict of style layers names to their weights.
        :param training_images: list of path of training images.
        :param style_image: style image.
        :param session: active tf.Session.
        :param discriminator: network that used for loss calculation.
        :param num_training_epochs: number of training epoches.
        :param batch_size: training batch size.
        :param content_loss_weight: weight of content loss.
        :param style_loss_weight: weight of style loss.
        :param total_variance_loss_weight: weight of total variance loss.
        :param learning_rate: learning rate.
        :param model_save_path: path to dir where store training models.
        :param saving_period: how often make model checkpoint.
        """

        self._input_shape = input_shape
        self.discriminator = discriminator
        self._sess = session

        # sort layers info
        self.CONTENT_LAYERS = collections.OrderedDict(
            sorted(content_layer_dict.items())
        )
        self.STYLE_LAYERS = collections.OrderedDict(
            sorted(style_layer_dict.items())
        )

        # input images
        self._train_images_list = training_images
        mod = len(training_images) % batch_size

        if mod > 0:
            self._train_images_list = self._train_images_list[:-mod]

        self._num_of_examples = len(self._train_images_list)
        self._style_image = style_image

        # parameters for optimization
        self._num_epochs = num_training_epochs
        self._content_weight = content_loss_weight
        self._style_weight = style_loss_weight
        self._total_variance_weight = total_variance_loss_weight
        self._learn_rate = learning_rate
        self._batch_size = batch_size
        self._check_period = saving_period

        # path for model to be saved
        self.save_path = model_save_path

        # image transform network
        self.transform = transform.Transform()

        # build graph for style transfer
        self._build_graph()

    def _build_losses(self):
        """
        Builds training losses.
        """

        total_style_loss = self._content_weight * self._build_style_loss()
        total_content_loss = self._style_weight * self._build_content_loss()

        total_variance_loss = compute_total_variation_loss(
            self.generated_image
        )
        total_variance_loss *= self._total_variance_weight

        self._total_loss = (total_style_loss + total_content_loss +
                            total_variance_loss)

        # add summary for each loss
        tf.summary.scalar('content_loss', total_content_loss)
        tf.summary.scalar('style_loss', total_style_loss)
        tf.summary.scalar('total_variance_loss', total_variance_loss)
        tf.summary.scalar('loss', self._total_loss)

    def _build_graph(self):
        """
        Builds training graph.
        """

        self._build_graph_inputs()
        self._build_graph_endpoints()
        self._build_losses()
        self._build_optimizer()

    def _build_graph_endpoints(self):
        """
        Build training graph endpoints.
        """

        self._build_content_features_endpoints()
        self._build_style_features_endpoints()
        self._build_generator_features_endpoints()

    def _build_graph_inputs(self):
        """
        Build training graph inputs.
        """

        input_shape = (
            self._batch_size,
            self._input_shape[0],
            self._input_shape[1],
            3
        )

        self.content_img_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=input_shape,
            name='content'
        )

        self.style_image_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=self._style_image.shape,
            name='style'
        )

        self.preprocessed_content_img = self.discriminator.preprocess_inputs(
            self.content_img_placeholder
        )

        self.preprocessed_style_image = self.discriminator.preprocess_inputs(
            self.style_image_placeholder
        )

    def _build_content_features_endpoints(self):
        """
        Build content image features endpoints.
        """

        content_subnet_endpoints = self.discriminator.build(
            self.preprocessed_content_img,
            scope='content'
        )

        self.content_image_features = {}
        for layer_name in self.CONTENT_LAYERS:
            features = content_subnet_endpoints[layer_name]
            self.content_image_features[layer_name] = features

    def _build_style_features_endpoints(self):
        """
        Build style image features endpoints.
        """

        style_subnet_endpoints = self.discriminator.build(
            self.preprocessed_style_image,
            scope='style'
        )

        self.style_image_features = {}
        for layer_name in self.STYLE_LAYERS:
            features = style_subnet_endpoints[layer_name]
            self.style_image_features[layer_name] = features

    def _build_generator_features_endpoints(self):
        """
        Build generated image features endpoints.
        """

        preprocessed_generator_inputs = transform.Transform.preprocess_inputs(
            inputs=self.content_img_placeholder
        )
        self.generated_image = self.transform.net(preprocessed_generator_inputs)

        preprocessed_generated_image = self.discriminator.preprocess_inputs(
            image=self.generated_image
        )
        self.generated_image_features = self.discriminator.build(
            input_tensor=preprocessed_generated_image,
            scope='mixed'
        )

    def _build_content_loss(self):
        """
        Build content loss.
        """

        total_content_loss = 0

        for layer_name in self.CONTENT_LAYERS.keys():
            generated_image_layer = self.generated_image_features[layer_name]
            original_image_layer = self.content_image_features[layer_name]

            layer_weight = self.CONTENT_LAYERS[layer_name]

            total_content_loss += layer_weight * compute_content_loss(
                content_image_features=original_image_layer,
                generate_image_features=generated_image_layer,
                scope=(layer_name + '/content_loss')
            )

        return total_content_loss

    def _build_style_loss(self):
        """
        Build style loss.
        """

        total_style_loss = 0

        for layer_name in self.STYLE_LAYERS.keys():
            layer_weight = self.STYLE_LAYERS[layer_name]

            generated_image_layer = self.generated_image_features[layer_name]
            style_image_layer = self.style_image_features[layer_name]

            total_style_loss += layer_weight * compute_style_loss(
                style_image_features=style_image_layer,
                generate_image_features=generated_image_layer,
                scope=(layer_name + '/style_loss')
            )

        return total_style_loss

    def _build_optimizer(self):
        """
        Build optimizer.
        """

        self._global_step_tensor = tf.train.get_or_create_global_step()

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._total_loss, trainable_variables)

        optimizer = tf.train.AdamOptimizer(self._learn_rate)
        self._train_op = optimizer.apply_gradients(
            grads_and_vars=zip(grads, trainable_variables),
            global_step=self._global_step_tensor,
            name='train_op'
        )

    def _initialize_summary_writer(self):
        """
        Initialize summary_writer.
        """

        self._merged_summary_op = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter(
            logdir=self.save_path,
            graph=tf.get_default_graph()
        )

    def _restore_model_if_exist(self):
        """
        Check is older checkpoint exist and restore trainable params.
        """

        checkpoint_exists = True

        try:
            ckpt_state = tf.train.get_checkpoint_state(self.save_path)
        except:
            tf.logging.warning('Cannot restore checkpoint from {}'.format(
                self.save_path
            ))
            checkpoint_exists = False

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.warning('No model to restore at {}'.format(
                self.save_path
            ))
            checkpoint_exists = False

        if checkpoint_exists:
            tf.logging.info('Loading checkpoint {}'.format(
                ckpt_state.model_checkpoint_path
            ))
            self._saver.restore(self._sess, ckpt_state.model_checkpoint_path)

    def _initialize_model_saver(self):
        """
        Initialize model saver.
        """

        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()

    def _fetch_next_input_batch(self, placeholder: np.ndarray) -> np.ndarray:
        """
        Fetch input batch.
        """

        begin_index = self._global_step * self._batch_size
        end_index = min((self._global_step + 1) * self._batch_size,
                        self._num_of_examples)
        for index, image_path in enumerate(self._train_images_list[begin_index:end_index]):
            image = utils.get_img(
                image_path,
                img_size=(self._input_shape[0], self._input_shape[1], 3)
            )
            placeholder[index] = image.astype(np.float32)

        return placeholder

    def train(self):
        """
        Train style transfer model.
        """

        self._initialize_summary_writer()
        self._initialize_model_saver()
        self._restore_model_if_exist()
        self._global_step = self._sess.run(self._global_step_tensor)

        maximum_number_iterations = self._num_of_examples * self._num_epochs
        maximum_number_iterations /= self._batch_size

        inputs_placeholder = np.empty(shape=self._input_shape,
                                      dtype=np.float32)

        while self._global_step < maximum_number_iterations:
            self._fetch_next_input_batch(placeholder=inputs_placeholder)

            iteration_start_time = time.time()

            (_, summary, total_loss, self._global_step) = self._sess.run(
                [self._train_op, self._merged_summary_op,
                 self._total_loss, self._global_step_tensor],
                feed_dict={
                    self.content_img_placeholder: inputs_placeholder,
                    self.style_image_placeholder: self._style_image
                })

            time_delta = time.time() - iteration_start_time

            tf.logging.info(msg='Iter - {} - Loss - {} - Step - {:.2f}s'.format(
                self._global_step, total_loss, time_delta
            ))

            self._summary_writer.add_summary(summary, self._global_step)

            if self._global_step % self._check_period == 0:
                self._saver.save(
                    sess=self._sess,
                    save_path=self.save_path + '/final.ckpt',
                    global_step=self._global_step
                )

        self._saver.save(self._sess, self.save_path + '/final.ckpt')
        self._summary_writer.close()
