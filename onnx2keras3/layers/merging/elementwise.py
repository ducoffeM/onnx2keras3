import keras
import keras.backend as K
from keras.layers import Multiply, Lambda
import logging
from onnx2keras3.layers.utils import is_numpy, ensure_tf_type
import numpy as np
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer


class PlusConstant(keras.layers.Layer):
    """Custom Keras Layer that adds a constant value to a Keras Tensor.
    This layer performs element-wise addition of a constant value to a Keras Tensor.
    """

    def __init__(self, constant: Union[float, ArrayLike], minus: bool = False, **kwargs):
        """
        Compute the result of (-1 * x + constant) or (x + constant), depending on the 'minus' parameter.
        Args:
            constant: The constant value to be added to the tensor.
            minus: The indicator for the operation to be performed:
                 - If minus equals 1, it computes (-1 * x + constant).
                 - If minus equals -1, it computes (x + constant).
        """
        super(PlusConstant, self).__init__(**kwargs)
        self.constant: Union[float, ArrayLike] = constant
        self.sign: int = 1
        if minus:
            self.sign = -1

    def call(self, inputs_):
        return self.sign * inputs_ + self.constant

    def get_config(self):
        config = super().get_config()
        config.update({"constant": self.constant, "sign": self.sign})
        return config

    def compute_output_spec(self, *args, **kwargs):

        return args[0]


class MulConstant(keras.layers.Layer):
    """Custom Keras Layer that multiply a constant value to a Keras Tensor.
    This layer performs element-wise multiplication of a constant value to a Keras Tensor.
    """

    def __init__(self, constant: Union[float, ArrayLike], **kwargs):
        """
        Compute the result of  x*constant.
        Args:
            constant: The constant value to be elementwise multiplied with the tensor.
        """
        super(MulConstant, self).__init__(**kwargs)
        self.constant: Union[float, ArrayLike] = constant

    def call(self, inputs_):
        return self.constant * inputs_

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "constant": self.constant,
            }
        )
        return config

    def compute_output_spec(self, *args, **kwargs):

        return args[0]


class DivConstant(keras.layers.Layer):
    """Custom Keras Layer that divide a constant value with a Keras Tensor.
    This layer performs element-wise division of a constant value and a Keras Tensor.
    """

    def __init__(self, constant: Union[float, ArrayLike], **kwargs):
        """
        Compute the result of  x*constant.
        Args:
            constant: The constant value to be elementwise multiplied with the tensor.
        """
        super(DivConstant, self).__init__(**kwargs)
        self.constant: Union[float, ArrayLike] = constant

    def call(self, inputs_):
        return self.constant / inputs_

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "constant": self.constant,
            }
        )
        return config

    def compute_output_spec(self, *args, **kwargs):

        return args[0]


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


def convert_elementwise_div(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an elementwise Div layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the elemntwise Div layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger("onnx2keras.e_div")
    lambda_layer: Layer
    output: Tensor
    input_0: Union[ArrayLike, Tensor] = inputs[0]
    input_1: Union[ArrayLike, Tensor] = inputs[1]

    if len(node.input) != 2:
        raise AttributeError("Number of inputs is not equal 2 for element-wise layer")

    if is_numpy(input_0) and is_numpy(input_1):
        logger.debug("Divide numpy arrays.")
        output = input_0 / input_1  # convert to a tensor !!!!
    elif is_numpy(input_1):
        lambda_layer = MulConstant(constant=1.0 / input_1, name=keras_name)
        output = lambda_layer(input_0)
    else:
        lambda_layer = DivConstant(constant=input_0, name=keras_name)
        output = lambda_layer(input_1)

    return output


def convert_elementwise_add(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an elementwise Add layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the elemntwise Add layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    output: Tensor
    input_0: Tensor = inputs[0]
    input_1: Tensor = inputs[1]

    if K.is_keras_tensor(input_0) and K.is_keras_tensor(input_1):
        output = keras.layers.Add()(inputs)
    elif K.is_keras_tensor(inputs[0]):
        output = PlusConstant(constant=inputs[1])(inputs[0])
    else:
        output = PlusConstant(constant=inputs[0])(inputs[1])

    return output


def convert_elementwise_mul(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an elementwise Mul layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the elemntwise Mul layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    input_0: Tensor = inputs[0]
    input_1: Tensor
    output: Tensor
    mul: Layer

    try:
        input_1 = inputs[1]
    except IndexError:
        input_1 = weights[0]

    if not K.is_keras_tensor(input_1):
        if len(input_0.shape) > len(input_1.shape):
            input_1: ArrayLike = np.expand_dims(input_1, -1)
        mul = MulConstant(constant=input_1, name=keras_name)
        output = mul(input_0)
        return output

    if not K.is_keras_tensor(input_0):
        input_0: ArrayLike = input_0.numpy()
        if len(input_1.shape) > len(input_0.shape):
            input_0 = np.expand_dims(input_0, -1)

        mul = MulConstant(constant=input_0, name=keras_name)
        # Lambda(lambda z: z*input_0, name=keras_name)
        output = mul(input_1)
        return output

    if data_format_keras != data_format_onnx:
        if len(input_0.shape) > len(input_1.shape):

            input_1 = K.expand_dims(input_1, -1)  # do a function !!!!
        elif len(input_0.shape) < len(input_1.shape):

            input_0 = K.expand_dims(input_0, -1)  # do a function !!!!

    try:
        mul = keras.layers.Multiply(name=keras_name)
        output = mul([input_0, input_1])

    except (IndexError, ValueError):
        logger.warning("Failed to use keras.layers.Multiply. Fallback to TF lambda.")
        lambda_layer: Layer = Multiply(name=keras_name)
        output = lambda_layer([input_0, input_1])

    return output


def convert_elementwise_sub(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an elementwise Subtract layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the elemntwise Subtract layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """

    logger = logging.getLogger("onnx2keras.sub")
    if len(inputs) != 2:
        raise AttributeError("Number of inputs is not equal 2 for element-wise layer")

    logger.debug("Convert inputs to Keras/TF layers if needed.")
    input_0: Tensor = inputs[0]
    input_1: Tensor = inputs[1]
    output: Tensor
    layer: Layer

    if K.is_keras_tensor(input_0) and K.is_keras_tensor(input_1):
        layer = keras.layers.Subtract()
        output = layer([input_0, input_1])
    elif K.is_keras_tensor(input_0):
        layer = PlusConstant(constant=-input_1)
        output = layer(input_0)

    else:
        layer = PlusConstant(constant=input_0, minus=True)
        output = layer(input_1)

    return output


def convert_min(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Min layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Min layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    if len(inputs) < 2:
        assert AttributeError("Less than 2 inputs for min layer.")

    layer: Layer = keras.layers.Minimum(name=keras_name)
    output: Tensor = layer(inputs)

    return output


def convert_max(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Max layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Max layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    if len(inputs) < 2:
        assert AttributeError("Less than 2 inputs for max layer.")

    layer: Layer = keras.layers.Maximum(name=keras_name)
    output: Tensor = layer(inputs)

    return output


def convert_mean(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts an Average/Mean layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Average/Mean layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    if len(inputs) < 2:
        assert AttributeError("Less than 2 inputs for max layer.")

    layer: Layer = keras.layers.Average(name=keras_name)
    output: Tensor = layer(inputs)

    return output
