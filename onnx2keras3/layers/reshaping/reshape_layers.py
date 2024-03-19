import keras
import keras.backend as K
import numpy as np
import logging
from onnx2keras3.layers.utils import is_numpy, ensure_tf_type, ensure_numpy_type
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer
from onnx2keras3.layers.reshaping.utils import Squeeze, Repeat

def convert_transpose(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a Transpose layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Transpose layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.transpose')
    perm:ArrayLike = params['perm']
    permute:Layer = keras.layers.Permute(perm[1:], name=keras_name)
    output:Tensor = permute(inputs[0])
    return output


def convert_shape(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a get_shape layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the get_shape layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.shape')
    input_0 = inputs[0]

    logger.debug('Actual shape:')
    logger.debug(np.array(input_0.shape))

    shapes:List[Union[int, None]] = []
    for i in input_0.shape:
        if i is not None:
            shapes.append(i)
        else:
            shapes.append(None)

    output:Tensor = np.array(shapes)
    return output


def convert_gather(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a gather layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the gather layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.gather')
    output:ArrayLike

    if is_numpy(inputs[0]) and is_numpy(inputs[1]):
        logger.debug('Gather from numpy array')

        if params['axis'] == 0:
            output = np.array(layers[node.input[0]][inputs[1]])
        elif params['axis'] == 1:
            output = np.array(layers[:, node.input[0]][inputs[1]])
        elif params['axis'] == 2:
            output = np.array(layers[:, :, node.input[0]][inputs[1]])
        elif params['axis'] == 3:
            output = np.array(layers[:, :, :, node.input[0]][inputs[1]])
        else:
            raise AttributeError('Can\'t gather by axis more than 3.')
    else:
        raise AttributeError('Can\'t gather from tf tensor.')

    return output


def convert_concat(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a Concatenate layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Concatenate layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    axis:int =params['axis']
    return keras.layers.Concatenate(axis=axis)(inputs)


def convert_reshape(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a Reshape layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Reshape layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.reshape')

    input_0:Tensor = inputs[0]
    input_1 : Tensor
    output: Tensor
    try:
        input_1:Tensor = inputs[1]
    except IndexError:
        input_1:ArrayLike = weights[0]

    if is_numpy(input_1):
        logger.debug('The second argument is numpy array.')
        if is_numpy(input_0):
            logger.debug('The first argument is numpy array. Apply np.reshape.')
            output = np.reshape(input_0, np.int32(input_1))
        else:
            if 'change_ordering' in params.keys() and params['change_ordering']:
                """
                raise NotImplementedError()
                
                # Fix critical issue with NHWC
                if input_1[0] is None and input_1[1] == -1:
                    logger.warning('!!! IMPORTANT INFORMATION !!!')
                    logger.warning('The target shape if [None, -1] that means flatten.')
                    logger.warning('But the target ordering is NHWC, so we cant simply perform flatten')
                    logger.warning('The layer will be converted as lambda with tf.transpose')
                    logger.warning('---')

                    def target_layer(x):
                        x = keras.ops.transpose(x, [0, 3, 1, 2])
                        return x

                    lambda_layer = keras.layers.Lambda(target_layer, name="%s_CHW" % keras_name)
                    layers[node_name] = lambda_layer(input_0)
                    lambda_func[keras_name] = target_layer
                else:
                    layers[node_name] = input_0
                """
                reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
                output = reshape(input_0)

            else:
                logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
                logger.debug('Target shape :')
                logger.debug(np.int32(input_1[1:]))

                if len(np.int32(input_1[1:])) == 1 and np.int32(input_1[1:])[0] == -1:
                    logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')
                    flatten = keras.layers.Flatten(name=keras_name)
                    output = flatten(input_0)
                else:
                    if data_format_keras!=data_format_onnx:
                        if len(input_1)==3:
                            input_1 = [input_1[i] for i in [0, 2, 1]]
                        if len(input_1)==4:
                            input_1 = [input_1[i] for i in [0, 3, 1, 2]] # [0, 3, 2, 1]

                    reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
                    output = reshape(input_0)

    else:
        raise AttributeError('Can\'t reshape dynamic size.')

    return output


def convert_unsqueeze(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts an Unsqueeze layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Unsqueeze layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.unsqueeze')

    if len(inputs) != 1:
        raise AttributeError('Number of inputs is not equal 1 for unsqueeze layer')
    
    input_0:Tensor = inputs[0]
    output:Tensor

    if is_numpy(input_0):
        logger.debug('Work with numpy types.')
        output = inputs[0]
        for axis in params['axes']:
            output = np.expand_dims(output, axis)
    else:

        if len(params['axes']) != 1:
            raise AttributeError('Number of axes is not equal 1. Cannot unsqueeze')

        axis=params['axes'][0]
        input_shape:List[Union[int, None]] = list(input_0.shape)
        input_shape = input_shape[:axis]+[1]+input_shape[axis:]
        lambda_layer:Layer = Reshape(input_shape[1:], name=keras_name)
        output = lambda_layer(inputs[0])
        return output


def convert_flatten(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a Flatten layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Flatten layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.flatten')

    if len(inputs) != 1:
        raise AttributeError('Number of inputs is not equal 1 for flatten layer')

    logger.debug('Convert inputs to Keras/TF layers if needed.')
    input_0 = inputs[0]

    if params['change_ordering']:
        lambda_layer:Layer = keras.layers.Permute([0, 3, 1, 2], name="%s_CHW" % keras_name)
        tensor_chw = lambda_layer(input_0)
        flatten = keras.layers.Flatten(name=keras_name)
        output:Tensor = flatten(tensor_chw)
    else:
        reshape = keras.layers.Reshape([-1], name=keras_name)
        output:Tensor = reshape(input_0)
    
    return output


def convert_squeeze(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a Squeeze layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Squeeze layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.squeeze')
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for squeeze layer.')

    input_0:Tensor = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    lambda_layer:Layer = Sqeeze(axis=params['axes'][0], name=keras_name)
    output:Tensor = lambda_layer(input_0)
    return output


def convert_expand(node:Node, inputs:List[Tensor], weights:List[ArrayLike], params:Dict[str, Any] , keras_name:str, data_format_onnx:DataFormat, data_format_keras:DataFormat)->Union[Tensor, List[Tensor]]:
    """
    Converts a Squeeze layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Squeeze layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    logger = logging.getLogger('onnx2keras.expand')
    if len(inputs) != 2:
        assert AttributeError('More than 2 input for expand layer.')

    input_0:Tensor = inputs[0]
    input_1:Tensor = ensure_numpy_type(inputs[1])

    lambda_layer:Layer = Repeat(shape=input_1, name=keras_name)
    output:Tensor = lambda_layer(input_0)
    return output
