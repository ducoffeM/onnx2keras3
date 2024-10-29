import keras
import logging
from onnx2keras3.layers.utils import is_numpy
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer
from .utils import ensure_numpy_type


def convert_gemm(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Convert Linear / GEMM layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Conv layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.gemm")

    W: Union[Tensor, ArrayLike]
    bias: Union[None, ArrayLike]
    has_bias: bool
    input_0: Tensor
    output: Tensor
    input_channels: int
    output_channels: int
    reshape: Layer
    dense: Layer
    reshaped_x: Tensor

    # Check if Bias available
    if len(weights) == 2:
        logger.debug("Conv with bias")
        has_bias = True
        if len(weights[0].shape)<len(weights[1].shape):
            W = ensure_numpy_type(weights[1])
            bias = ensure_numpy_type(weights[0])
        else:
            W = ensure_numpy_type(weights[0])
            bias = ensure_numpy_type(weights[1])
        logger.debug("Convert GEMM with bias.")
    elif len(weights) == 1:
        logger.debug("Conv without bias")
        has_bias = False
        W = ensure_numpy_type(weights[0])
        bias = None
        logger.debug("Convert GEMM without bias.")
    else:
        if len(inputs) >= 2:
            W = inputs[1]
        else:
            raise AttributeError("More than 3 or less than 2 weights")

    input_0: Tensor = inputs[0]

    # Linear can have additional flag to transpose weights
    if "transB" in params and params["transB"] == 1:
        logger.debug("Transposing W matrix.")
        W = W.transpose()

    # Estimate input/output neurons
    input_channels, output_channels = W.shape
    logger.debug("Input units %s, output units %s.", input_channels, output_channels)

    if is_numpy(W):
        bias_initializer: Union[str, Constant] = "zeros"
        if has_bias:
            bias_initializer = keras.initializers.Constant(bias)

        kernel_initializer = keras.initializers.Constant(W)
        dense = keras.layers.Dense(
            output_channels,
            name=keras_name,
            bias_initializer=bias_initializer,
            kernel_initializer=kernel_initializer,
            use_bias=has_bias,
        )

        # The first input - always X
        try:
            output = dense(input_0)
        except ValueError:
            reshape = keras.layers.Reshape([input_channels], name=keras_name + "_reshape")
            reshaped_x = reshape(input_0)
            output = dense(reshaped_x)

    else:
        output = keras.layers.Multiply()([input_0, W])

    return output
