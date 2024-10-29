from onnx2keras3.layers import *


# source of onnx operator:  https://github.com/onnx/onnx/blob/main/docs/Operators.md
def default_convert(**kwargs):
    raise NotImplementerError()


default_mapping_onnx2keras_classes = {
    "Conv": convert_conv,
    "ConvTranspose": convert_convtranspose,
    "Relu": convert_relu,
    "Elu": convert_elu,
    "LeakyRelu": convert_lrelu,
    "Sigmoid": convert_sigmoid,
    "Tanh": convert_tanh,
    "Selu": convert_selu,
    "Clip": convert_clip,
    "Exp": convert_exp,
    "Log": convert_log,
    "Softmax": convert_softmax,
    "PRelu": convert_prelu,
    "ReduceMax": convert_reduce_max,
    "ReduceSum": convert_reduce_sum,
    "ReduceMean": convert_reduce_mean,
    "Pow": convert_reduce_pow,
    "Slice": convert_slice,
    "Squeeze": convert_squeeze,
    "Expand": convert_expand,
    "Sqrt": convert_reduce_sqrt,
    "Split": convert_split,
    "Cast": convert_cast,
    "Floor": convert_floor,
    "Identity": convert_identity,
    "ArgMax": convert_argmax,
    "ReduceL2": convert_reduce_l2,
    "Max": convert_max,
    "Min": convert_min,
    "Mean": convert_mean,
    "Div": convert_elementwise_div,
    "Add": convert_elementwise_add,
    "Sum": convert_elementwise_add,
    "Mul": convert_elementwise_mul,
    "Sub": convert_elementwise_sub,
    "Gemm": convert_gemm,
    "MatMul": convert_gemm,
    "Transpose": convert_transpose,
    "BatchNormalization": convert_batchnorm,
    "InstanceNormalization": convert_instancenorm,
    "Dropout": convert_dropout,
    "LRN": convert_lrn,
    "MaxPool": convert_maxpool,
    "AveragePool": convert_avgpool,
    "GlobalAveragePool": convert_global_avg_pool,
    "Shape": convert_shape,
    "Gather": convert_gather,
    "Unsqueeze": convert_unsqueeze,
    "Concat": convert_concat,
    "Reshape": convert_reshape,
    "Pad": convert_padding,
    "Flatten": convert_flatten,
    "Upsample": convert_upsample,
    "Slice": convert_slice,
    "Cast": convert_cast,
    #"Resize": convert_resize,
}


def get_layer(op_type: str):

    return default_mapping_onnx2keras_classes[op_type]
