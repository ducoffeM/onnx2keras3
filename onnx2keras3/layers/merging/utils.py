import keras
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
        self.constant: Union[float, Tensor] = keras.ops.convert_to_tensor(constant)
        self.sign: int = 1
        if minus:
            self.sign = -1

    def call(self, inputs_):
        return self.sign * inputs_ + self.constant

    def get_config(self):
        config = super().get_config()
        config.update({"constant": self.constant, "sign": self.sign})
        return config

    def compute_output_shape(self, input_shape):

        return input_shape


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
        if len(constant.shape):
            self.constant: Union[float, Tensor] = keras.ops.convert_to_tensor(constant)
        else:
            self.constant: Union[float, Tensor] = constant

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

    def compute_output_shape(self, input_shape):

        return input_shape


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

        if len(constant.shape):
            self.constant: Union[float, Tensor] = keras.ops.convert_to_tensor(constant)
        else:
            self.constant: Union[float, Tensor] = constant

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

    def compute_output_shape(self, input_shape):

        return input_shape
