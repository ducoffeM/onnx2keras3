import keras
import logging
from onnx2keras3.layers.utils import ensure_tf_type, ensure_numpy_type


def convert_batchnorm(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert BatchNorm2d layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.batchnorm2d")

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    if len(node.input) == 5:
        weights = [
            ensure_numpy_type(layers[node.input[1]]),
            ensure_numpy_type(layers[node.input[2]]),
            ensure_numpy_type(layers[node.input[3]]),
            ensure_numpy_type(layers[node.input[4]]),
        ]
    elif len(node.input) == 3:
        weights = [ensure_numpy_type(layers[node.input[1]]), ensure_numpy_type(layers[node.input[2]])]
    else:
        raise AttributeError("Unknown arguments for batch norm")

    eps = params["epsilon"] if "epsilon" in params else 1e-05  # default epsilon
    momentum = params["momentum"] if "momentum" in params else 0.9  # default momentum

    if len(weights) == 2:
        logger.debug("Batch normalization without running averages")
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps, center=False, scale=False, weights=weights, name=keras_name
        )
    else:
        bn = keras.layers.BatchNormalization(axis=1, momentum=momentum, epsilon=eps, weights=weights, name=keras_name)

    layers[node_name] = bn(input_0)


def convert_instancenorm(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert InstanceNorm2d layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.instancenorm2d")

    raise NotImplementedError()


def convert_dropout(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Dropout layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.dropout")

    # In ONNX Dropout returns dropout mask as well.
    if isinstance(keras_name, list) and len(keras_name) > 1:
        keras_name = keras_name[0]

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    ratio = params["ratio"] if "ratio" in params else 0.0
    lambda_layer = keras.layers.Dropout(ratio, name=keras_name)
    layers[node_name] = lambda_layer(input_0)


def convert_lrn(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert LRN layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.LRN")
    logger.debug("LRN can't be tested with PyTorch exporter, so the support is experimental.")

    raise NotImplementedError()
