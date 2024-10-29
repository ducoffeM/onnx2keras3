import keras
import logging
from onnx2keras3.layers.utils import ensure_tf_type
from typing import List, Dict, Any, Union
from onnx2keras3.typing import Tensor, Node, WeightsOnnx, DataFormat, ArrayLike, Padding, Layer


def convert_padding(
    node: Node,
    inputs: List[Tensor],
    weights: List[ArrayLike],
    params: Dict[str, Any],
    keras_name: str,
    data_format_onnx: DataFormat,
    data_format_keras: DataFormat,
) -> Union[Tensor, List[Tensor]]:
    """
    Converts a Padding layer from ONNX to Keras.

    Args:
        node: The ONNX node representing the Padding layer.
        inputs: A list of Keras input tensors.
        weights: A list of the node's parameters.
        keras_name: The name of the layer in Keras.
        data_format_onnx: The data format used during ONNX conversion.
        data_format_keras: The data format used when building the Keras model.

    Returns:
        Tensor: The output tensor/s after applying the equivalent Keras layer to the inputs of the ONNX node.
    """
    # It's binary by-default
    logger = logging.getLogger("onnx2keras.padding")
    input_0: Tensor = inputs[0]
    pads = params["pads"]
    """
    Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. 
    For 2D input tensor, it is the number of pixels. pads should be a 1D tensor of shape [2 * num_axes] where num_axes refers to the number 
    of elements in the axes input or the input rank if axes are not provided explicitly. pads format should be: [x1_begin, x2_begin, …, x1_end, x2_end,…],
      where xi_begin is the number of pad values added at the beginning of axis axes[i] and xi_end, the number of pad values added at the end of axis axes[i].
    """
    if pads is None:
        raise TypeError("missing padding dimension")
    if max(params['pads'])==0 and max(params['pads'])==0:
        return input_0
    if len(pads) > 4:
        # source: KaidiXu onnx2pytorch
        # pads should be [0,0,pad_top,pad_left,0,0,pad_bottom,pad_right]
        assert pads[0] == pads[1] == pads[4] == pads[5] == 0
        pads = ((int(pads[3]), int(pads[7])), (int(pads[2]), int(pads[6])))

    dico_config = {"trainable": True, "padding": pads, "data_format": data_format_onnx}


    layer = keras.layers.ZeroPadding2D.from_config(dico_config)
    output = layer(input_0)
    return output
