import numpy as np
from xml.dom import minidom

## slightly modified nml parser originally written by nils

class NodeAttribute:

    def __init__(self, position, id, orientation, partner, identifier):

        self.position = position

        self.id = id

        self.orientation = orientation

        self.partner = partner

        self.identifier = identifier

def parse_attributes(xml_elem, parse_input):

    parse_output = []

    attributes = xml_elem.attributes

    for x in parse_input:

        try:

            parse_output.append(x[1](attributes[x[0]].value))

        except KeyError:

            parse_output.append(None)

    return parse_output

def from_node_elem_to_node(node_elem):

    [x, y, z, ID, radius, orientation, partner, identifier] = parse_attributes(
            node_elem, [
                ["x", float],
                ["y", float],
                ["z", float],
                ["id", int],
                ["radius", float],
                ["orientation", str],
                ["partner", int],
                ["identifier", int]
                ]
            )


    point = np.array([x, y, z])

    if orientation is not None:

        orientation = orientation.replace("[", " ").replace("]", " ")

        split = orientation.split(" ")

        rec_ori = []

        for i in split:

            try:

                rec_ori.append(float(i))

            except:

                continue

        orientation = np.array(rec_ori)

        for j in range(np.shape(orientation)[0]):

            orientation[j] = float(orientation[j])

    else:

        orientation = np.array([0., 0., 0.])


    return point, ID, np.array(orientation), partner, identifier

def parse_nml(in_file):
    doc = minidom.parse(in_file)

    annotation_elems = doc.getElementsByTagName("thing")

    node_dic = {}

    edge_list = []

    for annotation_elem in annotation_elems:

        node_elems = annotation_elem.getElementsByTagName("node")

        for node_elem in node_elems:

            node_attribute = NodeAttribute(*from_node_elem_to_node(node_elem))

            if node_attribute.id in node_dic:

                print('WARNING: ID already exists')

                break

            else:

                node_dic[node_attribute.id] = node_attribute

        edge_elems = annotation_elem.getElementsByTagName("edge")

        for edge_elem in edge_elems:

            (source_ID, target_ID) = parse_attributes(edge_elem, [["source", int], ["target", int]])

            edge_list.append(sorted([source_ID, target_ID]))

    return node_dic, edge_list
