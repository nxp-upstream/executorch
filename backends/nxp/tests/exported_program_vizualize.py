import random

from gvgen import GvGen
from torch.export import ExportedProgram


def exported_program_to_dot(exported_program: ExportedProgram, dot_file_name="graph.dot", show_tags=True,
                            show_arguments=True):
    """
    Generate dot file for tagged exported program.

    :param exported_program: Exported program with optional meta values: 'delegation_tag' and 'cluster'.
    :param dot_file_name: Produced .dot file name.
    :param show_tags: If True, nodes will be shown as a subcomponent of tag nodes.
    :param show_arguments: If True, node arguments will be shown in exported dot file.
    """
    graph = GvGen()

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
    for node in exported_program.graph.nodes:
        if "delegation_tag" in node.meta and show_tags:
            tag = node.meta["delegation_tag"]
            if tag not in delegation_tags:
                item = graph.newItem(tag)
                delegation_tags[tag] = item

    for node in exported_program.graph.nodes:
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
        if any(q in label for q in ["_quantize_per_tensor_", "_quantize_per_channel_"]):
            graph.propertyAppend(item, "shape", "invhouse")
        elif any(dq in label for dq in ["_dequantize_per_tensor_", "_dequantize_per_channel_"]):
            graph.propertyAppend(item, "shape", "house")
        else:
            graph.propertyAppend(item, "shape", "box")

        graph_items[node.name] = item

    # Add connections between nodes
    for node in exported_program.graph.nodes:
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
