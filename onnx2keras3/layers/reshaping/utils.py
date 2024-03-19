from typing import List, Dict, Any
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer

import keras
import keras.backend as K
import numpy as np
from keras.layers import Layer, Lambda


class Squeeze(Layer):
    def __init__(self, axis: int, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis: int = dtype

    def call(self, inputs_):
        return keras.ops.squeeze(inputs_, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_spec(self, *args, **kwargs):

        return keras.ops.squeeze(args[0], self.axis)


class Repeat(Layer):
    def __init__(self, shape, **kwargs):
        super(Repeat, self).__init__(**kwargs)
        self.shape = shape

    def call(self, inputs_):
        inputs_ = keras.backend.repeat_elements(inputs_, int(self.shape[1] // inputs_.shape[1]), 1)
        inputs_ = keras.backend.repeat_elements(inputs_, int(self.shape[2] // inputs_.shape[2]), 2)
        return inputs_

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_spec(self, *args, **kwargs):

        raise ValueError()
