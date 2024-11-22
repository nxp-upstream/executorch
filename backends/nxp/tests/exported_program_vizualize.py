import random

from gvgen import GvGen
from torch.export import ExportedProgram


def exported_program_to_dot(exported_program: ExportedProgram):
    """
    Generate dot file for tagged exported program.

    :param exported_program: Exported program with optional meta values: 'delegation_tag' and 'cluster'.
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
        if "delegation_tag" in node.meta:
            tag = node.meta["delegation_tag"]
            if tag not in delegation_tags:
                item = graph.newItem(tag)
                delegation_tags[tag] = item

    for node in exported_program.graph.nodes:
        if "delegation_tag" in node.meta:
            # Delegated node -> add color
            tag = node.meta["delegation_tag"]
            item = graph.newItem(node.name, delegation_tags[tag])

            graph.propertyAppend(item, "fillcolor", name_color(tag))
            graph.propertyAppend(item, "style", "filled")
        else:
            item = graph.newItem(node.name)

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
            graph.newLink(graph_items[node.name], graph_items[user.name])

    with open("graph.dot", "w") as f:
        graph.dot(f)
