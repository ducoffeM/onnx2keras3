import keras
import keras.backend as K
import logging
from onnx2keras3.layers.utils import is_numpy, ensure_tf_type, ensure_numpy_type
import numpy as np
from keras.layers import Layer, Lambda
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer
from onnx2keras3.layers.operation.utils import Clip, Log, Sum, Mean, Max, Split, Identity, Cast, Argmax, ReduceL2, Floor


# Handle python 2.7 import error
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def convert_clip(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Clip layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Clip layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.clip")

    if len(inputs) != 1:
        assert AttributeError("More than 1 input for clip layer.")

    input_0: Tensor = inputs[0]
    output: Tensor
    layer: Layer

    if params["min"] == 0:
        logger.debug("Using ReLU({0}) instead of clip".format(params["max"]))
        layer = keras.layers.ReLU(max_value=params["max"], name=keras_name)
    else:
        layer = Clip(vmin=params["min"], vmax=params["max"], name=keras_name)

    output = layer(input_0)
    return output


def convert_log(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Log activation layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Log activation layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.log")

    if len(inputs) != 1:
        assert AttributeError("More than 1 input for log layer.")

    input_0: Tensor = inputs[0]

    lambda_layer: Layer = Log(name=keras_name)
    output: Tensor = lambda_layer(input_0)
    return output


def convert_reduce_sum(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Reduce Sum layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Reduce Sum layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.sum")

    if len(inputs) != 1:
        assert AttributeError("More than 1 input for reduce sum layer.")

    input_0: Tensor = inputs[0]
    axis: Union[int, None]
    output: Tensor

    try:
        axis: int = params["axes"]
    except KeyError:
        if len(weights):
            axis = np.max(weights[0]).astype("int64")
        else:
            axis = None

    lambda_layer: Layer = Sum(axis=axis, name=keras_name)

    output = lambda_layer(input_0)
    return output


def convert_reduce_mean(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Reduce Mean layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Reduce Mean layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.mean")

    if len(node.input) != 1:
        assert AttributeError("More than 1 input for reduce mean layer.")

    input_0: Tensor = inputs[0]

    lambda_layer: Layer = Mean(axis=params["axes"], keepdims=(params["keepdims"] == 1), name=keras_name)
    output = lambda_layer(input_0)

    return output


def convert_reduce_max(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Reduce Max layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Reduce Max layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.max")

    if len(node.input) != 1:
        assert AttributeError("More than 1 input for reduce mean layer.")

    input_0: Tensor = inputs[0]

    lambda_layer: Layer = Max(axis=params["axes"], keepdims=(params["keepdims"] == 1), name=keras_name)
    output = lambda_layer(input_0)

    return output


def convert_reduce_pow(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Pow layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Pow layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.pow")

    input_0: Tensor = inputs[0]
    power: Any
    if len(inputs) > 1:
        power = inputs[1]
    elif len(weights):
        power = weights[0]
    else:
        raise AttributeError("Missing power in the inputs")

    lambda_layer: Layer = Pow(power=power, name=keras_name)
    output = lambda_layer(input_0)

    return output


def convert_reduce_sqrt(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Sqrt layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Sqrt layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.sqrt")

    input_0: Tensor = inputs[0]

    lambda_layer: Layer = Sqrt(name=keras_name)
    output = lambda_layer(input_0)

    return output


def convert_split(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a split layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Split layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.split")

    axis: int = params.get("axis", 0)
    splits: int

    if "split" in params.keys():
        splits = params["split"]
    else:
        splits = inputs[1]

    layer: Layer = Split(splits=splits[:-1], axis=axis, name=keras_name)  # TO DO: several layers
    output: Tensor = layer(inputs[0])

    return output


def convert_cast(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a cast layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the cast layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.cast")
    input_0: Tensor = inputs[0]
    output: Tensor

    if len(inputs) != 1:
        assert AttributeError("More than 1 input for cast layer.")

    if is_numpy(input_0):
        logger.debug("Cast numpy array")

        cast_map = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            9: np.bool,
            10: np.float16,
            11: np.double,
        }

        output = cast_map[params["to"]](inputs[0])
    else:

        def target_layer(x, dtype=params["to"]):
            cast_map = {
                1: K.float32,
                2: K.uint8,
                3: K.int8,
                5: K.int16,
                6: K.int32,
                7: K.int64,
                9: K.bool,
                10: K.float16,
                11: K.double,
            }
            return keras.ops.cast(x, cast_map[dtype])

        lambda_layer: Layer = Cast(dtype=params["to"], name=keras_name)
        output = lambda_layer(input_0)

    return output


def convert_floor(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a floor layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the floor layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.floor")

    if len(inputs) != 1:
        assert AttributeError("More than 1 input for floor layer.")

    input_0: Tensor = inputs[0]

    lambda_layer = Floor(name=keras_name)
    output: Tensor = lambda_layer(input_0)
    return output


def convert_identity(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Convert Identity layer
    """
    logger = logging.getLogger("onnx2keras.identity")

    if len(node.input) != 1:
        assert AttributeError("More than 1 input for itentity layer.")

    return inputs[0]


def convert_argmax(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an argmax layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the argmax layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.argmax")

    if len(node.input) != 1:
        assert AttributeError("More than 1 input for argmax layer.")

    input_0 = inputs[0]
    axis: Union[None, int] = params.get("axis", -1)

    lambda_layer: Layer = Argmax(name=keras_name)
    output: Tensor = lambda_layer(input_0)
    return output


def convert_reduce_l2(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a ReduceL2 layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the ReduceL2 layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.reduce_l2")

    if len(node.input) != 1:
        assert AttributeError("More than 1 input for argmax layer.")

    input_0 = inputs[0]
    axis: int = params.get("axis", -1)
    keepdims: int = params.get("keepdims", 0)

    lambda_layer: Layer = ReduceLp(axis=axis, keepdims=(keepdims == 1), name=keras_name)
    output: Tensor = lambda_layer(input_0)
    return output
