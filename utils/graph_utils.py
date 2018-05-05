from typing import Iterable

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

from nets.transformationnet import TransformationNet


def load_model_from_ckpt(sess: tf.Session,
                         model_path: str) -> tf.Session:
    """
    Load model from checkpoint.
    :param sess: Active TensorFlow session.
    :param model_path: model checkpoint path.
    :return: Active TensorFlow session containing the variables.
    """

    saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
    saver.restore(sess, model_path)

    return sess


def freeze_graph(sess: tf.Session,
                 output_node_name: str) -> tf.Graph:
    """
    Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    :param sess: Active TensorFlow session containing the variables.
    :param output_node_name: name of the result node in the graph.
    :return: GraphDef containing a simplified version of the original.
    """

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=[output_node_name]
    )

    return output_graph_def


def optimize_graphdef(input_graph_def: tf.GraphDef,
                      input_node_name: str,
                      output_node_name: str) -> tf.GraphDef:
    """
    Optimize input GraphDef.
    :param input_graph_def: GraphDef containing a network.
    :param input_node_name: GraphDef input node name.
    :param output_node_name: GraphDef output node name.
    :return: GraphDef containing a optimized version of the original.
    """

    optimized_graph_def = TransformGraph(
        input_graph_def=input_graph_def,
        inputs=[input_node_name],
        outputs=[output_node_name],
        transforms=["merge_duplicate_nodes",
                    "strip_unused_nodes",
                    "remove_device",
                    "fold_constants",
                    "flatten_atrous_conv",
                    "fold_batch_norms",
                    "fold_old_batch_norms",
                    "fuse_pad_and_conv",
                    "fuse_resize_pad_and_conv",
                    "sort_by_execution_order"]
    )

    return optimized_graph_def


# TODO: move to networks folder.
def create_transformation_network(
        sess: tf.Session,
        model_ckpt_path: str,
        input_node_name: str,
        desired_input_shape: Iterable[int]) -> tf.Graph:
    """
    Create transformation network from checkpoint.
    :param sess: Active TensorFlow session.
    :param model_ckpt_path: path where model checkpoint are stored.
    :param input_node_name: model input node name.
    :param desired_input_shape: transformation input shape.
    :return: tf.Graph with loaded model.
    """
    img_placeholder = tf.placeholder(tf.float32,
                                     shape=desired_input_shape,
                                     name=input_node_name)

    transformation_network = TransformationNet()
    transformation_network.build_transformation_net(img_placeholder)

    saver = tf.train.Saver()
    try:
        saver.restore(sess, model_ckpt_path)
    except:
        raise RuntimeError('Something wrong with checkpoint. '
                           'Check checkpoint path.')

    return sess.graph
