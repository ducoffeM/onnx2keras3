import keras
from keras.activations import softmax
import logging
from onnx2keras3.layers.utils import ensure_tf_type, ensure_numpy_type

from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike


def convert_relu(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a ReLU activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the ReLU activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.relu")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    if len(params):
        raise NotImplementedError()
        # keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0, **kwargs)
        relu = keras.layers.ReLU(name=keras_name)
    else:
        relu = keras.layers.Activation("relu", name=keras_name)

    return relu(inputs[0])


def convert_elu(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a ELU activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the ELU activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.elu")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    if len(params) == 0 or params["alpha"] == 1.0:
        elu = keras.layers.Activation("elu", name=keras_name)
    else:
        elu = keras.layers.ELU(alpha=params["alpha"], name=keras_name)

    return elu(inputs[0])


def convert_lrelu(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a LeakyReLU activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the LeakyReLU activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.lrelu")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    if len(params) == 0 or params["alpha"] == 1.0:
        lrelu = keras.layers.Activation("leaky_relu", name=keras_name)
    else:
        lrelu = keras.layers.LeakyReLU(negative_slope=params["alpha"], name=keras_name)

    return lrelu(inputs[0])


def convert_sigmoid(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Sigmoid activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Sigmoid activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.sigmoid")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    sigmoid = keras.layers.Activation("sigmoid", name=keras_name)

    return sigmoid(inputs[0])


def convert_tanh(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Tanh activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Tanh activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.tanh")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    tanh = keras.layers.Activation("tanh", name=keras_name)

    return tanh(inputs[0])


def convert_selu(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a SELU activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the SELU activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.selu")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    selu = keras.layers.Activation("selu", name=keras_name)

    return selu(inputs[0])


def convert_softmax(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Softmax activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Softmax activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.softmax")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    axis_ = params["axis"]
    if axis_ == -1:
        softmax = keras.layers.Activation("softmax", name=keras_name)
    else:
        softmax = keras.layers.Softmax(axis=axis_, name=keras_name)

    return softmax(inputs[0])


def convert_prelu(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a PReLU activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the PReLU activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    logger = logging.getLogger("onnx2keras.prelu")
    if len(inputs) != 2 and len(weights) == 0:
        assert AttributeError("Activation layer PReLU should have 2 inputs or weights.")

    # retrive weights
    if len(inputs) == 2:
        W = ensure_numpy_type(inputs[1])
    else:
        W = ensure_numpy_type(weights[0])

    if params["change_ordering"]:
        logger.warning("PRelu + change ordering needs to be fixed after TF graph is built.")
        logger.warning("It's experimental.")

    shared_axes = [2, 3]

    # for case when W.shape (n,). When activation is used for single dimension vector.
    shared_axes = shared_axes if len(W.shape) > 1 else None

    prelu = keras.layers.PReLU(weights=[W], shared_axes=shared_axes, name=keras_name)

    return prelu(inputs[0])


def convert_exp(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Exp activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Exp activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    logger = logging.getLogger("onnx2keras.softmax")
    if len(inputs) > 1:
        raise AttributeError("More than 1 input for an activation layer.")

    exp = keras.layers.Activation("exp", name=keras_name)

    return exp(inputs[0])
