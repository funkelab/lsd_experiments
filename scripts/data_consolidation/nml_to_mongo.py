import collections
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

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)

def get_neuron_id(in_file, prefix):

    in_file = os.path.basename(in_file)

    neuron_id = in_file.strip(prefix).strip('.nml')

    return int(neuron_id)

def convert_to_world(location, voxel_size):

    return (location * voxel_size)

if __name__ == '__main__':

    db_host = "mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke",
    db_name = 'zebrafinch_gt_skeletons_test_infinite_roi_delete_3'

    voxel_size = [9, 9, 20]

    files = glob.glob(os.path.join(sys.argv[1], '*.nml'))

    graph_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        nodes_collection='zebrafinch.nodes',
        edges_collection='zebrafinch.edges',
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'])

    roi = daisy.Roi(
    (-10e6, )*3,
    (20e6,)*3)

    graph = graph_provider[roi]

    n = []
    e = []

    test = []

    start = 0

    for nml_file in files:

        print('loading %s'%nml_file)

        nodes, edges = parse_nml(nml_file)

        print(nodes)

        length = len(nodes.keys())

        new_ids = list(range(start,start+length))

        #remap nodes start at zero
        node_map = {i:j for i,j in zip(nodes.keys(), new_ids)}
        new_nodes = {node_map[i]:j for i,j in nodes.items()}

        #remap edges
        new_edges = [[node_map[u],node_map[v]] for u,v in edges]

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
            for node_id, node in new_nodes.items()
        ])

        # for i,j in new_nodes.items():
            # n.append(j.position[2])

        node_ids = set(new_nodes.keys())

        #todo remap edges
        for u,v in new_edges:
            if u not in node_ids or v not in node_ids:
                raise RuntimeError(f"One of {u}, {v} not in list of nodes!")
            graph.add_edge(u, v)

        start += length
    # print(max(n))

    # # print(len(n), sum(n))

    # print(len(test), len(set(test)))

    # print([item for item, count in collections.Counter(test).items() if count>1])

    # print(graph.nodes())
    # print(graph.number_of_nodes())

    # print(len(e), sum(e))
    # graph.write_nodes(roi)
    # graph.write_edges(roi)


