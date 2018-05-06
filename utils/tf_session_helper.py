import tensorflow as tf


class TFSessionHandler(object):
    """
    TensorFlow Session handler.
    """
    def __init__(self, scope: str = ""):
        """
        :param scope: a name scope of model.
        """
        self.sess = tf.Session()
        self.scope = scope

    def __del__(self):
        self.sess.close()

    def _load_protobuf(self, protobuf_path: str) -> tf.Graph:
        """
        Load model from protobuf into graph.
        :param protobuf_path: path to model protobuf
        :return: graph with loaded model.
        """

        tf_graph = tf.Graph()

        with tf_graph.as_default():
            tf_graph_def = tf.GraphDef()
            with tf.gfile.GFile(protobuf_path, 'rb') as protobuf_file:
                serialized_graph = protobuf_file.read()
                tf_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(tf_graph_def, name=self.scope)

        tf.contrib.graph_editor.copy(tf_graph, self.sess.graph)

        return self.sess.graph
