import daisy
import numpy as np
import pymongo
import sys
import os
import glob
from xml.dom import minidom
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import BulkWriteError

from nml_parser import *

def get_neuron_id(in_file, prefix):

    in_file = os.path.basename(in_file)

    neuron_id = in_file.strip(prefix).strip('.nml')

    return int(neuron_id)

def convert_to_world(location, voxel_size):

    # nml starts with 1 and is in voxel space

    return ((location - 1) * voxel_size)

if __name__ == '__main__':

    db_host = "mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke",
    db_name = 'zebrafinch_gt_skeletons'

    voxel_size = [9, 9, 20]

    files = glob.glob(os.path.join(sys.argv[1], '*.nml'))

    graph_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        nodes_collection='nodes',
        edges_collection='edges',
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'])

    roi = daisy.Roi(
    (0, 0, 0),
    (114000, 98217, 95976))

    graph = graph_provider[roi]

    nodes_db = []
    edges_db = []

    for nml_file in files:

        print(nml_file)

        nodes, edges = parse_nml(nml_file)

        neuron_id = get_neuron_id(nml_file, 'test_set_skeleton')

        graph.add_nodes_from([
            (node_id, {
                'x': convert_to_world(node.position[0], voxel_size[0]),
                'y': convert_to_world(node.position[1], voxel_size[1]),
                'z': convert_to_world(node.position[2], voxel_size[2]),
                'neuron_id': neuron_id,
                'type': 'neuron'
                }
            )
            for node_id, node in nodes.items()
        ])


        for u,v in edges:
            graph.add_edge(u, v)


    graph.write_nodes(roi)
    graph.write_edges(roi)


