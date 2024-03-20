"""Description ...
"""

import onnx
import numpy as np
from onnx import numpy_helper
from onnx2keras3.utils import (
    onnx_node_attributes_to_dict,
    get_id,
    get_children_by_name,
    get_parents_by_name,
    get_layer_name,
)
import keras
import keras.backend as K
from keras.layers import Input
from onnx2keras3.layers.convert import get_layer

from typing import List, Dict
from onnx2keras3.typing import Tensor, Node, Graph, WeightsOnnx, DataFormat, ArrayLike, KerasModel, ModelOnnx


def iter_node(
    current_nodes: List[Node],
    dict_input_tensor: Dict[str, List[Tensor]],
    onnx_nodes: List[Node],
    constant_nodes: List[Node],
    onnx_input: List[Node],
    onnx_input_names: List[str],
    onnx_weights: List[WeightsOnnx],
) -> List[Node]:
    """
    Given the nodes that have been previously processed,
    this function removes from the current nodes those
    whose output has already been computed,
    and adds their children to the list of current nodes.

    Args:
        current_nodes: Nodes whose output has not yet been computed.
        dict_input_tensor: Output tensors from nodes that have been previously computed.
        constant_nodes: Nodes containing constant values.
        onnx_input: ONNX input nodes of the ONNX graph.
        onnx_weights: Weights of the ONNX graph.

    Returns:
        Next nodes whose output could be processed.
    """

    next_nodes: List[Node] = []
    new_nodes: List[Node] = []
    dict_update_tensor: Dict[str, List[Tensor]] = {}
    previous_nodes: List[str] = list(dict_input_tensor.keys())
    # at least one current_node should be treated

    for k, current_node in enumerate(current_nodes):

        if get_id(current_node) in previous_nodes:

            raise ValueError(
                "this {}-th node {} has been treated before, are we looping ?".format(k, get_id(current_node))
            )

        else:

            # check that we have at our disposal all the inputs to compute the output of the current node
            ##parents_nodes:List[str] = dico_children_2_parents[get_id(current_node)]
            parents_nodes: List[Node] = get_parents_by_name(
                current_node.input, onnx_nodes + constant_nodes, onnx_input, onnx_input_names
            )
            if len(parents_nodes) == 0:
                raise ValueError("no parents found for node {}".format(get_id(node)))

            # check that every parent's outputs has been computed
            check_input: bool = min([get_id(parent_node) in dict_input_tensor.keys() for parent_node in parents_nodes])

            if check_input:
                # retrieve input_tensor for keras
                input_tensors: List[Tensor] = []
                for parent_node in parents_nodes:
                    p_output_tensors: List[Tensor] = dict_input_tensor[get_id(parent_node)]
                    if len(p_output_tensors) > 1:
                        # multiple output, select the one used for the current node

                        output_names: List[str] = parent_node.output
                        input_names: List[str] = current_node.input
                        indices: List[int] = list(
                            np.where([output_name in input_names for output_name in output_names])[0]
                        )
                        input_tensors += [p_output_tensors[i] for i in indices]
                        # [np.where(input_name in ) for input_name in current_node.input if input_name in parent_node.output]
                    else:
                        input_tensors += p_output_tensors
                # we can convert current node to its Keras version and compute its output
                # TO DO: KERAS CONVERTER of current node
                convert_2_layer: Converter = get_layer(current_node.op_type)

                # (node, inputs, weights, params, node_name, keras_name)
                # retrieve weights
                weights: List[ArrayLike] = [
                    numpy_helper.to_array(onnx_weight)
                    for onnx_weight in onnx_weights
                    if onnx_weight.name in current_node.input
                ]
                # onnx_node_attributes_to_dict
                # retrieve params
                params_node: Dict[str, Any] = onnx_node_attributes_to_dict(current_node.attribute)
                data_format_onnx: DataFormat = "channels_first"
                data_format_keras: DataFormat = "channels_first"

                output_tensor: Union[List[None], Tensor, List[Tensor]] = [None]
                if not max([input_ is None for input_ in input_tensors]):
                    keras_name: str = get_layer_name(current_node)
                    try:
                        output_tensor = convert_2_layer(
                            current_node,
                            input_tensors,
                            weights,
                            params_node,
                            keras_name,
                            data_format_onnx,
                            data_format_keras,
                        )
                    except TypeError:
                        import pdb

                        pdb.set_trace()
                    if not isinstance(output_tensor, list):
                        output_tensor = [output_tensor]
                else:
                    output_tensor = [None]
                # current_node's output as heebn computed, we can store it
                dict_update_tensor[get_id(current_node)] = output_tensor  # compute it with Keras

                # now we can add the children nodes from current_node in the next iteration
                children_names: List[str] = current_node.output
                children: List[Node] = get_children_by_name(children_names, onnx_nodes)
                for child in children:
                    if get_id(child) in previous_nodes:
                        raise ValueError(
                            "this child {} has been treated before, are we looping ?".format(get_id(current_node))
                        )

                new_nodes += children
            else:
                # not enough inputs at this stage to compute current_node's output
                next_nodes.append(current_node)

    next_nodes += new_nodes
    # clean nex_nodes to avoid duplicates
    indices: List[int] = list(np.unique([get_id(next_node) for next_node in next_nodes], return_index=True)[1])
    next_nodes_unique: List[Node] = [next_nodes[i] for i in indices]
    # remove next_nodes that have been treated to avoid looping
    next_nodes: List[Node] = next_nodes_unique

    if len(new_nodes) == 0 and len(next_nodes):
        raise ValueError("empy set of nodes")
    dict_input_tensor.update(dict_update_tensor)
    return next_nodes


def onnx_to_keras(filename: str) -> KerasModel:
    """
    Convert ONNX graph to Keras model format
    Args:
        filename: path to the onnx file

    Returns:
        Keras model
    """
    onnx_model: ModelOnnx = onnx.load(filename)
    onnx.checker.check_model(onnx_model)

    # graph:Graph  = onnx_model.graph
    # model's parameters
    onnx_weights: List[WeightsOnnx] = [elem for elem in onnx_model.graph.initializer]
    # model's inputs
    onnx_inputs: List[Node] = [elem for elem in onnx_model.graph.input]
    # model's output
    onnx_outputs: List[Node] = [elem for elem in onnx_model.graph.output]
    # nodes of the onnx graph
    onnx_nodes: List[Node] = [elem for elem in onnx_model.graph.node]
    # model's output names
    # onnx_output_names: List[str] = [get_id(onnx_output) for onnx_output in onnx_outputs]
    # model's input names
    onnx_input_names: List[str] = [get_id(onnx_input) for onnx_input in onnx_inputs]
    # model's parameters names
    # onnx_weights_names: List[str] = [get_id(onnx_weight) for onnx_weight in onnx_weights]
    constant_nodes: List[Node] = [elem for elem in onnx_nodes if elem.op_type == "Constant"]
    # remove constant_nodes from onnx_nodes
    constant_nodes_names: List[str] = [get_id(e) for e in constant_nodes]
    onnx_nodes = [n for n in onnx_nodes if get_id(n) not in constant_nodes_names]

    dico_input_tensor: Dict[str, List[Tensor]] = {}
    for onnx_input, input_name in zip(onnx_inputs, onnx_input_names):
        input_shape: List[int] = [i.dim_value for i in onnx_input.type.tensor_type.shape.dim][1:]
        input_i: Tensor = Input(shape=input_shape, name=input_name)
        dico_input_tensor[get_id(onnx_input)] = [input_i]

    # add constant tensors
    for constant_node in constant_nodes:
        params_node: Dict[str, Any] = onnx_node_attributes_to_dict(constant_node.attribute)
        values: Any = params_node["value"]
        if not isinstance(values, list):
            values = [values]
        dico_input_tensor[get_id(constant_node)] = values

    # init current_nodes
    is_next_node: List[bool] = [
        max([input_name in onnx_node.input for input_name in onnx_input_names]) for onnx_node in onnx_nodes
    ]
    current_nodes: List[Node] = [onnx_nodes[i] for i in np.where(is_next_node)[0]]

    for _ in range(len(onnx_nodes) + 1):
        current_nodes = iter_node(
            current_nodes, dico_input_tensor, onnx_nodes, constant_nodes, onnx_inputs, onnx_input_names, onnx_weights
        )
        if len(current_nodes) == 0:
            break

    # retrieve inputs
    model_inputs: List[Tensor] = []
    for onnx_input in onnx_inputs:
        model_inputs += dico_input_tensor[get_id(onnx_input)]

    model_outputs: List[Tensor] = []
    for onnx_output in onnx_outputs:
        nodes: List[Node] = [node for node in onnx_nodes if onnx_output.name in node.output]
        for node in nodes:
            model_outputs += dico_input_tensor[get_id(node)]

    model: KerasModel = keras.models.Model(model_inputs, model_outputs)

    return model
