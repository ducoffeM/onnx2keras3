from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Callable
from typing_extensions import Literal

import numpy.typing

import keras
import onnx

# create extra types for readability


""" Type for Onnx objects """
ModelOnnx = onnx.onnx_ml_pb2.ModelProto
node_type = onnx.onnx_ml_pb2.NodeProto #node_type
node_value_type = onnx.onnx_ml_pb2.ValueInfoProto #node_value_type
Node = Union[node_type, node_value_type]
Graph = onnx.onnx_ml_pb2.GraphProto #graph_type
WeightsOnnx = onnx.onnx_ml_pb2.TensorProto #tensor_type
Attributes = Any
Attribute = Any

ArrayLike = numpy.typing.ArrayLike


"""Type for any tensor """
Tensor = Union[keras.KerasTensor, ArrayLike]

"""Type for data format"""
DataFormat = Literal["channels_first", "channels_last"]

"""Type for padding"""
Padding = Literal["same", "valid"]

"""Type for Converter"""
Converter = Callable

"""Type for Keras Model and Layers"""
KerasModel = keras.models.Model
Layer = keras.layers.Layer





