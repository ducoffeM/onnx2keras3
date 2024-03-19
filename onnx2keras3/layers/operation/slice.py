import keras
import keras.backend as K
import logging
from onnx2keras3.layers.utils import is_numpy
import numpy as np
from onnx2keras3.layers.operation.utils import Slice

from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer


def convert_slice(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Slicing layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Slicing layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    logger = logging.getLogger("onnx2keras.slice")

    data: Tensor
    inputs = inputs + weights

    starts: Union[List[int], Tensor]
    ends: Union[List[int], Tensor]
    axes: Union[List[int], Tensor]
    steps: Union[List[int], Tensor]

    # Check if optional parameters are available (axes, steps)
    if len(inputs) == 5:
        data = inputs[0]
        starts = inputs[1]
        ends = inputs[2]
        axes = inputs[3]
        steps = inputs[4]

    elif len(inputs) < 3:
        raise KeyError("expected at least three arguments but only got {}".format(len(node.input)))

    elif len(inputs) == 4:
        data = inputs[0]
        starts = inputs[1]
        ends = inputs[2]
        axes = inputs[3]
        steps = [0]

    else:
        data = inputs[0]
        starts = inputs[1]
        ends = inputs[2]
        axes = [0]
        steps = [0]

    if K.is_tensor(data):
        slice_ = Slice(axis=axes, starts=starts, ends=ends, steps=steps, name=keras_name)
        output: Tensor = slice_(data)
    else:
        output: Tensor = inputs[0][starts[0] : ends[0]]

    return output
