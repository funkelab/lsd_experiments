import numpy as np
import pymongo
import sys
import os
import glob
from xml.dom import minidom

'''

The following class and first 4 functions courtesy of Nils so i didnt have to
rewrite an nml parser. Given an nml file, `parse_nml` will return a node
dictionary and an edge list. The node dictionary contains the id, position, and
some other meta info for each node in the file. The edge list contains the
source and target nodes consituting an edge for each edge in the file. From the
node dict and edge list we create mongodb documents in the same schema as the
synful database `calyx_catmaid.nodes` and `calyx_catmaid.edges` collections. We
do this for every nml file in a directory.

'''

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

def get_neuron_id(in_file, prefix):

    in_file = os.path.basename(in_file)

    neuron_id = in_file.strip(prefix).strip('.nml')

    return int(neuron_id)

def create_collection(db, collection):

    if collection not in db.list_collection_names():
        collection = db[collection]
    else:
        collection = db[collection]

    return collection

def create_mongo_node_document(
        db,
        neuron_id,
        node_id,
        node_x,
        node_y,
        node_z,
        collection_name='nodes'):

    collection = create_collection(db, collection_name)

    document = {
            'neuron_id': neuron_id,
            'id': node_id,
            'x': node_x,
            'y': node_y,
            'z': node_z,
            'type': 'neuron'
    }

    print(document)

    collection.insert(document)

def create_mongo_edge_document(
        db,
        source,
        target,
        collection_name='edges'):

    collection = create_collection(db, collection_name)

    document = {
            'source': source,
            'target': target
    }

    print(document)

    collection.insert(document)

def convert_to_world(location, voxel_size):

    return int(location*voxel_size)

if __name__ == '__main__':

    db_host = "mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke",
    db_name = 'zebrafinch_gt_skeletons'

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    voxel_size = [9, 9, 20]

    files = glob.glob(os.path.join(sys.argv[1], '*.nml'))

    for nml_file in files:

        print(nml_file)

        nodes, edges = parse_nml(nml_file)

        neuron_id = get_neuron_id(nml_file, 'test_set_skeleton')

        for node_id, node in nodes.items():
            create_mongo_node_document(
                    db,
                    neuron_id,
                    node_id,
                    convert_to_world(node.position[0], voxel_size[0]),
                    convert_to_world(node.position[1], voxel_size[1]),
                    convert_to_world(node.position[2], voxel_size[2])
            )

        for u,v in edges:
            create_mongo_edge_document(db, u, v)
