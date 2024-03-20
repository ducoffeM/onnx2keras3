import onnx
from onnx import numpy_helper
import numpy as np
from typing import List, Dict, Any
from onnx2keras3.typing import Tensor, Node, Graph, WeightsOnnx, DataFormat, Attributes, Attribute, ArrayLike


def get_id(node: Node) -> str:
    """id of a node, either its ONNX name of python identifier if empty name

    Args:
        node: The ONNX node

    Returns:
        unique identifier
    """
    if len(node.name):
        return node.name
    else:
        return "{}".format(id(node))


# def assess_node_inputs(node_inputs_names:List[str], graph_inputs_names:List[str], graph_params_names:List[str])-> bool:
#
#    # check that all input of the node is an input of the graph or a parameter (from graph_initializer)
#    return len(node_inputs_names) and min([node_input_name in graph_inputs_names+graph_params_names for node_input_name in node_inputs_names])


def get_child_by_name(node_name: str, onnx_nodes: List[Node]) -> List[Node]:
    """Retrieve onnx nodes that are children of the onnx node with id node_name

    Args:
        node_name: id of ONNX node
        onnx_nodes: List of ONNX nodes in the ONNX graph

    Returns:
        ONNX nodes that are children of the specified node.
    """
    indices = np.where([node_name in n.input for n in onnx_nodes])[0]
    return [onnx_nodes[index] for index in indices]


def get_children_by_name(node_names: List[str], onnx_nodes: List[Node]) -> List[Node]:
    """Retrieve onnx nodes that are children of the onnx nodes with ids node_names

    Args:
        node_names: ids of ONNX node
        onnx_nodes: List of ONNX nodes in the ONNX graph

    Returns:
        ONNX nodes that are children of the specified nodes.
    """
    nodes: List[Node] = []
    for node_name in node_names:
        nodes += get_child_by_name(node_name, onnx_nodes)

    # clean  duplicates
    _, indices = np.unique([get_id(node) for node in nodes], return_index=True)
    nodes_unique = [nodes[i] for i in indices]
    return nodes_unique


def get_parent_by_name(
    node_input: str, onnx_nodes: List[Node], onnx_input: List[Node], onnx_input_names: List[str]
) -> List[Node]:
    """Retrieve ONNX nodes that are parents of the ONNX node using node_input as names of its inputs.

    Args:
        node_input: names of parents nodes
        onnx_nodes: ONNX nodes in the ONNX graph
        onnx_input: ONNX nodes that are input of the ONNX graph
        onnx_input_names: names of ONNX graph's inputs

    Returns:
        ONNX nodes that are parents of the specified node.
    """
    indices = np.where([node_input in n.output for n in onnx_nodes])[0]
    if len(indices) == 0 and node_input in onnx_input_names:
        # check if node_input is an onnx_input
        indices = np.where([node_input in onnx_input_names])[0]
        return [onnx_input[i] for i in indices]
    return [onnx_nodes[i] for i in indices]


def get_parents_by_name(
    node_inputs: List[str], onnx_nodes: List[Node], onnx_input: List[Node], onnx_input_names: List[str]
) -> List[Node]:
    """Retrieve ONNX nodes that are parents of the ONNX nodes using node_inputs as names of their inputs

    Args:
        node_input: names of parents nodes
        onnx_nodes: ONNX nodes in the ONNX graph
        onnx_input: ONNX nodes that are input of the ONNX graph
        onnx_input_names: names of ONNX graph's inputs

    Returns:
        ONNX nodes that are parents of the specified nodes.
    """
    nodes = []
    for node_input in node_inputs:
        nodes += get_parent_by_name(node_input, onnx_nodes, onnx_input, onnx_input_names)

    # clean  duplicates
    _, indices = np.unique([get_id(node) for node in nodes], return_index=True)
    nodes_unique = [nodes[i] for i in np.sort(indices)]  # sort indices to avoid ordering issue when parsing the graph
    return nodes_unique


def onnx_node_attributes_to_dict(args: Attributes) -> Dict[str, Any]:
    """Parse ONNX attributes to Python dictionary

    Args:
        args: ONNX attributes object

    Returns:
        Python dictionary
    """

    def onnx_attribute_to_dict(onnx_attr: Attribute) -> Any:
        """Parse ONNX attributes to Python dictionary

        Args:
            onnx_attr: ONNX attribute

        Returns:
            Python data type
        """
        if onnx_attr.HasField("t"):
            return numpy_helper.to_array(getattr(onnx_attr, "t"))

        for attr_type in ["f", "i", "s"]:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ["floats", "ints", "strings"]:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))

    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


# def retrieve_values(onnx_values:)->Dict[str, ArrayLike]:
#
#    return dict([(node_v.name, numpy_helper.to_array(node_v)) for node_v in onnx_values])


def get_layer_name(node: Node) -> str:
    if len(node.name):
        keras_name = node.name
    else:
        keras_name = node.output[0].split("/")[-1]

    if "/" in keras_name:
        keras_name = "_".join(keras_name.split("/"))[1:]

    return keras_name
