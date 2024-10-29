import keras
import logging
from onnx2keras3.layers.utils import ensure_tf_type

from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer


def convert_maxpool(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a MaxPooling layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the MaxPool layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.maxpool")
    input_0: Tensor = inputs[0]
    kernel_shape: ArrayLike = params["kernel_shape"]
    stride_shape: ArrayLike = params["strides"]

    if "auto_pad" in params:

        
        pad = params["auto_pad"].decode('latin1').lower().split('_')[0]
        if len(kernel_shape)==2:
            pooling: Layer = keras.layers.MaxPooling2D(pool_size=kernel_shape, strides=stride_shape, padding=pad, name=keras_name, data_format=data_format_keras)  
        else:
            pooling: Layer = keras.layers.MaxPooling3D(pool_size=kernel_shape, strides=stride_shape, padding=pad, name=keras_name, data_format=data_format_keras)  

            #input_0 = padding_layer(input_0)
        output: Tensor = pooling(input_0)

        return output   
    


    pads: ArrayLike = params["pads"] if "pads" in params else [0, 0, 0, 0, 0, 0]
    pad: Padding = "valid"

    if (
        all([shape % 2 == 1 for shape in kernel_shape])
        and all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))])
        and all([shape == 1 for shape in stride_shape])
    ):
        pad = "same"
        logger.debug("Use `same` padding parameters.")
    else:
        logger.warning("Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.")
        import pdb; pdb.set_trace()
        padding_name = keras_name + "_pad"
        if len(kernel_shape) == 2:
            padding = None

            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))

            if padding is not None:
                padding_layer: Layer = keras.layers.ZeroPadding2D(
                    padding=padding, name=padding_name, data_format=data_format_keras
                )
                input_0 = padding_layer(input_0)
        else:  # 3D padding
            padding_layer: Layer = keras.layers.ZeroPadding3D(
                padding=pads[: len(stride_shape)], name=padding_name, data_format=data_format_keras
            )
            input_0 = padding_layer(input_0)
    if len(kernel_shape) == 2:
        pooling: Layer = keras.layers.MaxPooling2D(
            pool_size=kernel_shape, strides=stride_shape, padding=pad, name=keras_name, data_format=data_format_keras
        )
    else:
        pooling: Layer = keras.layers.MaxPooling3D(
            pool_size=kernel_shape, strides=stride_shape, padding=pad, name=keras_name, data_format=data_format_keras
        )

    output: Tensor = pooling(input_0)
    return output


def convert_avgpool(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an AveragePooling layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the AveragePooling layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.avgpool")

    input_0 = inputs[0]

    kernel_shape: ArrayLike = params["kernel_shape"]
    stride_shape: ArrayLike = params["strides"]

    pads: ArrayLike = params["pads"] if "pads" in params else [0, 0, 0, 0, 0, 0]
    pad: Padding = "valid"

    if (
        all([shape % 2 == 1 for shape in kernel_shape])
        and all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))])
        and all([shape == 1 for shape in stride_shape])
    ):
        pad = "same"
        logger.debug("Use `same` padding parameters.")
    else:
        logger.warning("Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.")
        padding_name: str = keras_name + "_pad"
        if len(kernel_shape) == 2:
            padding_layer = keras.layers.ZeroPadding2D(
                padding=pads[: len(stride_shape)], name=padding_name, data_format=data_format_keras
            )
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[: len(stride_shape)], name=padding_name, data_format=data_format_keras
            )
        input_0 = padding_layer(input_0)
    if len(kernel_shape) == 2:
        pooling: Layer = keras.layers.AveragePooling2D(
            pool_size=kernel_shape, strides=stride_shape, padding=pad, name=keras_name, data_format="channels_first"
        )
    else:
        pooling: Layer = keras.layers.AveragePooling3D(
            pool_size=kernel_shape, strides=stride_shape, padding=pad, name=keras_name, data_format="channels_first"
        )

    output: Tensor = pooling(input_0)
    return output


def convert_global_avg_pool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert GlobalAvgPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger("onnx2keras.global_avg_pool")

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

    global_pool = keras.layers.GlobalAveragePooling2D(data_format="channels_first", name=keras_name)
    input_0 = global_pool(input_0)

    def target_layer(x):
        return keras.backend.expand_dims(x)

    logger.debug("Now expand dimensions twice.")
    lambda_layer1 = keras.layers.Lambda(target_layer, name=keras_name + "_EXPAND1")
    lambda_layer2 = keras.layers.Lambda(target_layer, name=keras_name + "_EXPAND2")
    input_0 = lambda_layer1(input_0)  # double expand dims
    layers[node_name] = lambda_layer2(input_0)
    lambda_func[keras_name + "_EXPAND1"] = target_layer
    lambda_func[keras_name + "_EXPAND2"] = target_layer
