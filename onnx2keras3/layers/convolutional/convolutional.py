import keras
import keras.backend as K
import logging
from onnx2keras3.layers.utils import ensure_tf_type, ensure_numpy_type
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer


def convert_conv(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Conv layer from ONNX to Keras.

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
    logger = logging.getLogger("onnx2keras.conv")
    W: ArrayLike
    bias: Union[None, ArrayLike]
    has_bias: bool

    out_channels: int
    channels_per_group: int
    dimension: int
    height: int
    width: int

    if len(weights) == 2:
        logger.debug("Conv with bias")
        has_bias = True
        W = ensure_numpy_type(weights[0])
        bias = ensure_numpy_type(weights[1])

    elif len(weights) == 1:
        logger.debug("Conv without bias")
        has_bias = False
        W = ensure_numpy_type(weights[0])
        bias = None

    else:
        raise NotImplementedError("Not implemented")

    input_0: Tensor = inputs[0]
    n_groups: Any = params["group"] if "group" in params else 1
    dilation: Any = params["dilations"][0] if "dilations" in params else 1
    pads: Any = params["pads"] if "pads" in params else [0, 0, 0]
    strides: Any = params["strides"] if "strides" in params else [1, 1, 1]
    padding_value: Padding = "valid"
    output: Tensor

    if len(W.shape) == 5:  # 3D conv
        logger.debug("3D convolution")
        padding_value: Padding = "valid"
        if pads[0] > 0 or pads[1] > 0 or pads[2] > 0:

            padding_value = "same"
            # logger.debug('Paddings exist, add ZeroPadding layer')
            # padding_name:str = keras_name + '_pad'
            # padding_layer:Layer = keras.layers.ZeroPadding3D(
            #    padding=(pads[0], pads[1], pads[2]),
            #    name=padding_name, data_format=data_format_keras
            # )
            # input_0 = padding_layer(input_0)

        out_channels, channels_per_group, dimension, height, width = W.shape
        W = W.transpose(2, 3, 4, 1, 0)

        conv: Layer = keras.layers.Conv3D(
            filters=out_channels,
            kernel_size=(dimension, height, width),
            strides=(strides[0], strides[1], strides[2]),
            padding=padding_value,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation,
            bias_initializer="zeros",
            kernel_initializer="zeros",
            name=keras_name,
            groups=n_groups,
        )
        conv.kernel = W
        if has_bias:
            conv.bias = bias
        conv.built = True
        output: Tensor = conv(input_0)

    elif len(W.shape) == 4:  # 2D conv
        logger.debug("2D convolution")
        padding_value: Padding = "valid"

        if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
            # padding = (pads[0], pads[1])
            padding_value = "same"

        elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
            # padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            padding_value = "same"

        W = W.transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape
        in_channels: int = channels_per_group * n_groups

        if n_groups == in_channels and n_groups != 1:
            logger.debug("Number of groups is equal to input channels, use DepthWise convolution")
            W = W.transpose(0, 1, 3, 2)

            conv = keras.layers.DepthwiseConv2D(
                kernel_size=(height, width),
                strides=(strides[0], strides[1]),
                padding=padding_value,
                use_bias=has_bias,
                activation=None,
                depth_multiplier=1,
                weights=weights,
                dilation_rate=dilation,
                bias_initializer="zeros",
                kernel_initializer="zeros",
                name=keras_name,
                data_format=data_format_keras,
            )
            conv.kernel = W
            if has_bias:
                conv.bias = bias
            conv.built = True
            output = conv(input_0)

        elif n_groups != 1:
            logger.debug("Number of groups more than 1, but less than number of in_channel, use group convolution")

            raise NotImplementedError()

        else:
            conv = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=(height, width),
                strides=(strides[0], strides[1]),
                padding=padding_value,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation,
                bias_initializer="zeros",
                kernel_initializer="zeros",
                name=keras_name,
                data_format=data_format_keras,
            )
            conv.kernel = W
            if has_bias:
                conv.bias = bias
            conv.built = True
            output = conv(input_0)
            return output
    else:
        # 1D conv
        W = W.transpose(2, 1, 0)
        width, channels, n_filters = W.shape
        if pads[0] > 0:
            padding_value = "same"

        conv = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=width,
            strides=strides[0],
            padding=padding_value,
            use_bias=has_bias,
            data_format=data_format_keras,
            dilation_rate=dilation,
            name=keras_name,
        )

        output = conv(input_0)

    return output


def convert_convtranspose(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a ConvTranspose layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the ConvTranspose layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.convtranpose")

    has_bias: bool
    W: ArrayLike
    b: Union[None, ArrayLike]

    height: int
    width: int
    n_filters: int
    channels: int
    pads: List[int]

    conv: Layer
    output: Tensor

    if len(weights) == 2:
        logger.debug("Conv with bias")
        has_bias = True
        W = ensure_numpy_type(weights[0])
        bias = ensure_numpy_type(weights[1])

    elif len(weights) == 1:
        logger.debug("Conv without bias")
        has_bias = False
        W = ensure_numpy_type(weights[0])
        bias = None

    else:
        raise NotImplementedError("Not implemented")

    input_0: Tensor = inputs[0]
    n_groups: Any = params["group"] if "group" in params else 1
    dilation: Any = params["dilations"][0] if "dilations" in params else 1
    pads: Any = params["pads"] if "pads" in params else [0, 0]
    strides: Any = params["strides"] if "strides" in params else [1, 1]

    if len(W.shape) == 5:  # 3D conv
        raise NotImplementedError("Not implemented")

    elif len(W.shape) == 4:  # 2D conv
        W = W.transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if n_groups > 1:
            raise AttributeError("Cannot convert ConvTranspose2d with groups != 1")

        if dilation > 1:
            raise AttributeError("Cannot convert ConvTranspose2d with dilation_rate != 1")

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=strides,
            padding="valid",
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation,
            bias_initializer="zeros",
            kernel_initializer="zeros",
            name=keras_name,
            data_format=data_format_keras,
        )
        conv.kernel = W
        if has_bias:
            conv.bias = bias
        conv.built = True

        if "output_shape" in params and "pads" not in params:
            logger.debug("!!!!! Paddings will be calculated automatically !!!!!")
            pads = [
                strides[0] * (int(input_0.shape[2]) - 1) + 0 + (height - 1) * dilation - params["output_shape"][0],
                strides[1] * (int(input_0.shape[3]) - 1) + 0 + (height - 1) * dilation - params["output_shape"][1],
            ]

        output = conv(input_0)

        if "output_padding" in params and (params["output_padding"][0] > 0 or params["output_padding"][1] > 0):
            raise AttributeError("Cannot convert ConvTranspose2d with output_padding != 0")

        if pads[0] > 0:
            logger.debug("Add cropping layer for output padding")
            assert len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1])

            crop: Layer = keras.layers.Cropping2D(pads[:2], name=keras_name + "_crop", data_format=data_format_keras)
            output = crop(output)
    else:
        raise AttributeError("Layer is not supported for now")

    return output
