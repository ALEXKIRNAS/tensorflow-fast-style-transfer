import numpy as np
import tensorflow as tf

import style_transfer_trainer
import utils.data_utils as utils
import vgg19
from utils.arg_parse_helpers import TrainArgsParser


def main():
    args_parser = TrainArgsParser()
    args = args_parser()

    vgg_net = vgg19.VGG19(args.vgg_model_weights_path)

    training_images_paths = utils.get_files(args.dataset_path)
    style_image = utils.load_image(args.style)

    # create a map for content layers info
    CONTENT_LAYERS = {}
    for layer, weight in zip(args.content_layers,args.content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    # create a map for style layers info
    STYLE_LAYERS = {}
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
        STYLE_LAYERS[layer] = weight

    # open session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # build the graph for train
    trainer = style_transfer_trainer.StyleTransferTrainer(
        session=sess,
        content_layer_ids=CONTENT_LAYERS,
        style_layer_ids=STYLE_LAYERS,
        content_images=training_images_paths,
        style_image=style_image,
        net=vgg_net,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        learn_rate=args.learn_rate,
        save_path=args.output,
        check_period=args.checkpoint_every,
        test_image=args.test,
        max_size=args.max_size
    )
    # launch the graph in a session
    trainer.train()

    # close session
    sess.close()


if __name__ == '__main__':
    main()
