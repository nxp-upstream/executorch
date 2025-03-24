import random
import warnings

from gvgen import GvGen
from torch.export import ExportedProgram
from torch.fx import Graph, GraphModule


def exported_program_to_dot(exported_program_or_graph: ExportedProgram | Graph | GraphModule,
                            dot_file_name="graph.dot", show_tags=True, show_arguments=False):
    warnings.warn("This function is now deprecated. Use program_or_graph_to_dot() instead.",
                  DeprecationWarning)
    program_or_graph_to_dot(exported_program_or_graph, dot_file_name=dot_file_name,
                            show_tags=show_tags, show_arguments=show_arguments)


def program_or_graph_to_dot(exported_program_or_graph: ExportedProgram | Graph | GraphModule,
                            dot_file_name="graph.dot", show_tags=True, show_arguments=False):
    """
    Generate dot file for tagged exported program, fx.Graph or GraphModule.

    :param exported_program_or_graph: Exported program, fx.Graph or GraphModule with optional node's meta
        values: 'delegation_tag' and 'cluster'.
    :param dot_file_name: Produced .dot file name.
    :param show_tags: If True, nodes will be shown as a subcomponent of tag nodes.
    :param show_arguments: If True, node arguments will be shown in exported dot file.
    """
    graph = GvGen()

    if isinstance(exported_program_or_graph, ExportedProgram):
        fx_graph = exported_program_or_graph.graph
    elif isinstance(exported_program_or_graph, Graph):
        fx_graph = exported_program_or_graph
    elif isinstance(exported_program_or_graph, GraphModule):
        fx_graph = exported_program_or_graph.graph
    else:
        print(f"Unsupported type of visualized program: '{type(exported_program_or_graph)}'.")
        exit(1)

    def name_color(string):  # pseudo-randomization function
        h = hash(string)  # hash string and int together
        if h < 0:  # ensure positive number
            h = h * -1
        random.seed(h)  # set the seed to use for randomization
        r = int(random.random() * 255)
        g = int(random.random() * 255)
        b = int(random.random() * 255)
        return '#%02x%02x%02x' % (r, g, b)

    graph_items = {}
    delegation_tags = {}

    # Find tags (parent objects)
    for node in fx_graph.nodes:
        if "delegation_tag" in node.meta and show_tags:
            tag = node.meta["delegation_tag"]
            if tag not in delegation_tags:
                item = graph.newItem(tag)
                delegation_tags[tag] = item

    for node in fx_graph.nodes:
        item_text = [node.name]

        if show_arguments and len(node.args) > 0:
            for i, arg in enumerate(node.args):
                arg_text = f"arg{i}: {str(arg)}"
                arg_text = arg_text[:45] + (arg_text[45:] and '..')
                item_text.append(arg_text)

            max_entry_len = len(max(item_text, key=len))
            item_text.insert(1, "-" * max_entry_len)

        if "delegation_tag" in node.meta and show_tags:
            # Delegated node -> add color
            tag = node.meta["delegation_tag"]
            item = graph.newItem("\n".join(item_text), delegation_tags[tag])

            graph.propertyAppend(item, "fillcolor", name_color(tag))
            graph.propertyAppend(item, "style", "filled")
        else:
            item = graph.newItem("\n".join(item_text))

        label = graph.propertyGet(item, "label")
        if "cluster" in node.meta:
            graph.propertyAppend(item, "label", label + "\n QDQ Cluster: " + node.meta["cluster"])

        # Change shape of node for (de)quantize and rest of nodes
        if any(dq in node.name for dq in ["dequantize_per_tensor_", "dequantize_per_channel_"]):
            graph.propertyAppend(item, "shape", "house")
        elif any(q in node.name for q in ["quantize_per_tensor_", "quantize_per_channel_"]):
            graph.propertyAppend(item, "shape", "invhouse")
        else:
            graph.propertyAppend(item, "shape", "box")

        graph_items[node.name] = item

    # Add connections between nodes
    for node in fx_graph.nodes:
        for user in node.users:
            link = graph.newLink(graph_items[node.name], graph_items[user.name])

            label = ""
            if "val" in node.meta:
                tensor = node.meta["val"]
                if isinstance(tensor, tuple):
                    tensor = tensor[0]  # Fake tensor
                label = f"  ({list(tensor.shape)} | {tensor.dtype})"

            graph.propertyAppend(link, "label", label)

    with open(dot_file_name, "w") as f:
        graph.dot(f)
