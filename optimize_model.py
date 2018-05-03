import os
import click
import tempfile

import tensorflow as tf

from utils import graph_utils


@click.command()
@click.option('--model_ckpt_path',
              help='Path to trained model.',
              default='./model/final.ckpt')
@click.option('--input_node_name',
              help='Name of input node.',
              default='content')
@click.option('--output_node_name',
              help='Name of output node.',
              default='styled_image')
@click.option('--desired_input_shape',
              help='Comma separated input shape.',
              default='1,360,640,3')
@click.option('--output_model_path',
              help='Path where to save optimized model.',
              default='./model/optimized_model.pb')
def main(model_ckpt_path: str,
         input_node_name: str,
         output_node_name: str,
         desired_input_shape: str,
         output_model_path: str):

    desired_input_shape = [
        int(value)
        for value in desired_input_shape.split(',')
    ]

    if len(desired_input_shape) != 4:
        raise ValueError('Input shape must be 4 dimensional.')

    sess = tf.Session()

    graph_utils.create_transformation_network(
        sess=sess,
        model_ckpt_path=model_ckpt_path,
        input_node_name=input_node_name,
        desired_input_shape=desired_input_shape
    )

    tmp_model_prefix = 'model'
    save_folder = tempfile.gettempdir()
    save_path = os.path.join(save_folder, tmp_model_prefix)

    saver = tf.train.Saver()
    saver.save(sess=sess, save_path=save_path, global_step=0)

    tf.reset_default_graph()

    graph_utils.load_model_from_ckpt(
        sess=sess,
        model_path=save_path + '-0'
    )

    frozen_graph_def = graph_utils.freeze_graph(
        sess=sess,
        output_node_name=output_node_name
    )

    optimized_graph_def = graph_utils.optimize_graphdef(
        input_graph_def=frozen_graph_def,
        input_node_name=input_node_name,
        output_node_name=output_node_name
    )

    save_model_dir = os.path.dirname(output_model_path)

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    tf.train.write_graph(
        graph_or_graph_def=optimized_graph_def,
        logdir='.',
        name=output_model_path,
        as_text=False
    )


if __name__ == '__main__':
    main()
