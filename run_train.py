import tensorflow as tf

import utils.data_utils as utils
from nets import vgg19
from trainers import style_transfer_trainer
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

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    trainer = style_transfer_trainer.StyleTransferTrainer(
        input_shape=(256, 256),
        session=sess,
        content_layer_dict=CONTENT_LAYERS,
        style_layer_dict=STYLE_LAYERS,
        training_images=training_images_paths,
        style_image=style_image,
        discriminator=vgg_net,
        num_training_epochs=args.num_epochs,
        batch_size=args.batch_size,
        content_loss_weight=args.content_weight,
        style_loss_weight=args.style_weight,
        total_variance_loss_weight=args.total_variance_weight,
        learning_rate=args.learn_rate,
        model_save_path=args.output,
        saving_period=args.checkpoint_every
    )

    trainer.train()
    sess.close()


if __name__ == '__main__':
    main()
