import torch.nn as nn
import pytest


@pytest.fixture(params=list(range(1)))
def n(request):
    return request.param


@pytest.fixture(params=["torch"])
def backend(request):
    return request.param


@staticmethod
def get_activation(n):
    if n == 0:
        return nn.ReLU()
    else:
        return nn.ReLU()


@staticmethod
def get_onnx_filename():
    return "test_torch.onnx"


@staticmethod
def get_input_shape():
    return (1, 3, 224, 224)
