import torch.nn as nn
import random
import numpy as np
import pytest
import onnx

import torch
from numpy.testing import assert_almost_equal

import os

import sys

sys.path.append("../../onnx2keras3")
from onnx2keras3.convert import onnx_to_keras


def test_activation():
    """
    class Layer(nn.Module):

        def __init__(self):
            super(Layer, self).__init__()
            function_torch = helpers.get_function(n)
            self.func = function_torch

        def forward(self, x):
            y = self.func(x)
            return y

    input_np = np.random.uniform(0, 1, helpers.get_input_shape())
    filename_onnx = helpers.get_onnx_filename()
    input_torch = torch.Tensor(input_np)
    model_torch = Layer()
    torch.onnx.export(model_torch,                                # model being run
                  input_torch,    # model input (or a tuple for multiple inputs)
                  filename_onnx)

    onnx_model = onnx.load(filename_onnx)
    onnx.checker.check_model(onnx_model)
    model_keras = onnx_to_keras(filename_onnx)

    output_torch = model_torch(input_torch).detach().numpy()
    output_keras = model_keras(input_torch).detach().numpy()

    # remove filename_onnx
    os.remove(filename_onnx)
    assert_almost_equal(output_torch, output_keras)
    """
