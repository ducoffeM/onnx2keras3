import keras
import numpy as np
import logging
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer


def convert_upsample(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an Upsample layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Upsample layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    logger = logging.getLogger("onnx2keras.upsample")
    logger.warning("!!! EXPERIMENTAL SUPPORT (upsample) !!!")

    if "scales" in params:
        # for opset version - 7
        if len(inputs) != 1:
            raise AttributeError("Unsupported number of inputs")
        scale: Union(tuple, List[int]) = np.uint8(params["scales"][-2:])
    elif len(inputs) != 1:
        # for opset version - 9+
        # Upsample since opset version 9 uses input[1] as 'scales' instead of attributes.
        scale: Union(tuple, List[int]) = np.uint8(inputs[1][-2:])
    else:
        scale: Union(tuple, List[int]) = np.uint8(weights[0][-2:])

    if params["mode"].decode("utf-8") != "nearest":
        logger.error("Cannot convert non-nearest upsampling.")
        raise AssertionError("Cannot convert non-nearest upsampling")

    upsampling = keras.layers.UpSampling2D(
        size=scale, name=keras_name, data_format=data_format_keras, interpolation=params["mode"].decode("utf-8")
    )

    return upsampling(inputs[0])
