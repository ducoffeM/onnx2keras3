import numpy as np
import keras
import logging
from onnx2keras3.layers.utils import ensure_tf_type, ensure_numpy_type
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike


def convert_batchnorm(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Convert BatchNorm2d layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the BatchNorm2d layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """


    logger = logging.getLogger("onnx2keras.batchnorm2d")
    input_0 = ensure_tf_type(inputs[0], name="%s_const" % keras_name)

    eps = params["epsilon"] if "epsilon" in params else 1e-05  # default epsilon
    momentum = params["momentum"] if "momentum" in params else 0.9  # default momentum
    
    weights_keras:List[keras.Variable]=[]
    if len(weights)==2 and len(node.input)-1==4:
        input_names = [inp.name for inp in inputs]
        weight_index = np.where([e =='weight' for e in input_names])[0][0]
        bias_index = np.where([e =='bias' for e in input_names])[0][0]
        print(weight_index, bias_index)
        gamma = inputs[weight_index].numpy()
        beta = inputs[bias_index].numpy()

        running_mean = weights[-1]
        running_var = weights[0]
    else:
        if len(weights)==4:
            gamma, beta, running_mean, running_var = weights
        else:
            gamma, beta, running_mean, running_var = weights

    if len(node.input)-1 == 2:
        logger.debug("Batch normalization without running averages")
        bn = keras.layers.BatchNormalization(axis=1, momentum=momentum, epsilon=eps, center=False, scale=False, name=keras_name)
        _ = bn(input_0)
    else:
        bn = keras.layers.BatchNormalization(axis=1, momentum=momentum, epsilon=eps, name=keras_name)
        _ = bn(input_0)
        bn.moving_mean.assign(running_mean)
        bn.moving_variance.assign(running_var)

    # set weights
    bn.gamma.assign(gamma)
    bn.beta.assign(beta)

    output = bn(input_0)

    return bn(input_0)






def convert_batchnorm_old(node, params, layers, lambda_func, node_name, keras_name, data_format_onnx, data_format_keras):
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
    print('bachtnorm')

    import pdb; pdb.set_trace()
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
