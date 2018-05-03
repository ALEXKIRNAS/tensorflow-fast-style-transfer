import argparse
import os
from abc import abstractmethod


class ArgParserHelper(object):
    def __init__(self):
        self._args = None

    def __call__(self):
        self._fetch_args()
        self._check_args()
        return self._args

    @abstractmethod
    def _check_args(self):
        pass

    @abstractmethod
    def _fetch_args(self):
        pass


class TrainArgsParser(ArgParserHelper):
    """
    Train parameters parser.
    """
    def _fetch_args(self):
        desc = ("TensorFlow implementation of " 
                "'Image Style Transfer' using convolutional "
                "neural networks.")

        parser = argparse.ArgumentParser(description=desc)

        parser.add_argument(
            '--vgg_model_weights_path',
            type=str,
            default='./data/weights/imagenet-vgg-verydeep-19.mat',
            help='The directory where the pre-trained model was saved.',
        )

        parser.add_argument(
            '--dataset_path',
            type=str,
            default='./data/train2014',
            help='The directory where dataset are stored.',
        )

        parser.add_argument(
            '--style',
            type=str,
            default='./data/style/wave.jpg',
            help='Style image path.',
        )

        parser.add_argument(
            '--output',
            type=str,
            default='./model',
            help='Path to folder, where store resulting model.',
        )

        parser.add_argument(
            '--_content_weight',
            type=float,
            default=7.5,
            help='Weight of content loss.'
        )

        parser.add_argument(
            '--_style_weight',
            type=float,
            default=500,
            help='Weight of style loss.'
        )

        parser.add_argument(
            '--_total_variance_weight',
            type=float,
            default=200,
            help='Weight of total variance loss.'
        )

        parser.add_argument(
            '--content_layers',
            nargs='+',
            type=str,
            default=['relu4_2'],
            help='List of VGG19 layers which used for content loss calculation.'
        )

        parser.add_argument(
            '--style_layers',
            nargs='+',
            type=str,
            default=['relu1_1',
                     'relu2_1',
                     'relu3_1',
                     'relu4_1',
                     'relu5_1'],
            help='List of VGG19 layers which used for style loss calculation.'
        )

        parser.add_argument(
            '--content_layer_weights',
            nargs='+',
            type=float,
            default=[1.0],
            help='Content per-layer weights.'
        )

        parser.add_argument(
            '--style_layer_weights',
            nargs='+',
            type=float,
            default=[.2, .2, .2, .2, .2],
            help='Style per-layer weights.'
        )

        parser.add_argument(
            '--_learn_rate',
            type=float,
            default=1e-3,
            help='Learning rate.'
        )

        parser.add_argument(
            '--_num_epochs',
            type=int,
            default=2,
            help='The number of epochs to run.'
        )

        parser.add_argument(
            '--_batch_size',
            type=int,
            default=4,
            help='Batch size.'
        )

        parser.add_argument(
            '--checkpoint_every',
            type=int,
            default=1000,
            help='Saving checkpoint frequency (as num of iterations).'
        )

        parser.add_argument(
            '--test',
            type=str,
            default=None,
            help='Test image path.'
        )

        parser.add_argument(
            '--max_size',
            type=int,
            default=None,
            help='The maximum width or height of input images.'
        )

        self._args = parser.parse_args()
        return self._args

    def _check_args(self):
        if not os.path.exists(self._args.vgg_model_weights_path):
            FileNotFoundError('File {} not found.'.format(
                self._args.vgg_model_weights_path
            ))

        expected_model_size_in_kb = os.path.getsize(
            self._args.vgg_model_weights_path
        )

        if abs(expected_model_size_in_kb - 534904783) > 10:
            raise FileExistsError('File {} have different size that expected. '
                                  'Please check your VGG19 model file.'.format(
                self._args.vgg_model_weights_path
            ))

        if not os.path.exists(self._args.dataset_path):
            raise FileNotFoundError('Folder {} does not exist.'.format(
                self._args.trainDB_path
            ))

        if not os.path.exists(self._args.style):
            raise FileNotFoundError('File {} not found.'.format(
                self._args.style
            ))

        if not os.path.exists(self._args.output):
            os.mkdir(self._args.output)

        if self._args.content_weight <= 0:
            raise ValueError('Content weight must be positive.')

        if self._args.style_weight <= 0:
            raise ValueError('Style weight must be positive.')

        if self._args.tv_weight <= 0:
            raise ValueError('Total variance weight must be positive.')

        content_weights_len = len(self._args.content_layer_weights)
        if len(self._args.content_layers) != content_weights_len:
            raise ValueError('Content layers and their weights must be same '
                             'length.')

        if len(self._args.style_layers) != len(self._args.style_layer_weights):
            raise ValueError('Style layers and their weights info must be same '
                             'length.')

        if self._args.learn_rate <= 0:
            raise ValueError('Learning rate must be positive.')

        if self._args.num_epochs < 1:
            raise ValueError('Number of epochs must be at least 1.')

        if self._args.batch_size < 1:
            raise ValueError('Batch size must be at least 1.')

        if self._args.checkpoint_every < 1:
            raise ValueError('Checkpoint saving period must be at least 1.')

        if self._args.test and not os.path.exists(self._args.test):
            raise FileNotFoundError('There is no %s' % self._args.test)

        if self._args.max_size and self._args.max_size < 0:
            raise ValueError('The maximum width or height of input image must '
                             'be positive')

        return self._args


class TestArgsParser(ArgParserHelper):
    """
    Test parameters parser.
    """
    def _fetch_args(self):
        desc = ("Tensorflow implementation of 'Perceptual Losses for " 
                "Real-Time Style Transfer and Super-Resolution'")
        parser = argparse.ArgumentParser(description=desc)

        parser.add_argument(
            '--style_model',
            type=str,
            default='./models/wave.ckpt',
            help='Style model checkpoint.',
        )

        parser.add_argument(
            '--content_image_path',
            type=str,
            default='./data/content/female_knight.jpg',
            help='File path of content image.',
        )

        parser.add_argument(
            '--output',
            type=str,
            default='./result.jpg',
            help='File path of resulting image.',
        )

        parser.add_argument(
            '--max_size',
            type=int,
            default=None,
            help='The maximum width or height of input images'
        )

        self._args = parser.parse_args()
        return self._args

    def _check_args(self):
        is_exist_index_file = os.path.exists(self._args.style_model + '.index')
        is_exist_meta_file = os.path.exists(self._args.style_model + '.meta')
        is_exist_data_file = os.path.exists(self._args.style_model +
                                            '.data-00000-of-00001')

        is_exist_model = all([
            is_exist_index_file, is_exist_meta_file, is_exist_data_file
        ])

        if not is_exist_model:
            raise FileNotFoundError('There is no {} model.'.format(
                self._args.style_model
            ))

        if not os.path.exists(self._args.content):
            raise FileNotFoundError('File {} not exist.'.format(
                self._args.content_image_path
            ))

        if self._args.max_size and self._args.max_size < 0:
            raise ValueError('The maximum width or height of input image '
                             'must be positive')

        return self._args
