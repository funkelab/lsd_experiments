import collections
import daisy
import glob
import json
import numpy as np
import os
import pymongo
import sys
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import BulkWriteError

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)

def get_neuron_id(in_file):

    in_file = os.path.basename(in_file)

    neuron_id = in_file.strip('.json')

    return int(neuron_id)

def convert_to_world(location, voxel_size):

    return (location * voxel_size)

if __name__ == '__main__':

    db_host = "mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke",
    db_name = 'zebrafinch_gt_skeletons_new_gt_9_9_20_delete_this'

    voxel_size = [9, 9, 20]
    # voxel_size = [10, 10, 20]

    # files = glob.glob(os.path.join(sys.argv[1], '*.json'))

    files = [sys.argv[1]]

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

    start = 0

    for json_file in files:

        print('loading %s'%json_file)

        with open(json_file, 'r') as f:
            skel = json.load(f)

            nodes = skel['nodes']
            edges = skel['links']

            length = len(nodes)
            new_ids = list(range(start, start + length))

            node_map = {i['id']: j for i,j in zip(nodes, new_ids)}
            edge_list = [[i['source'], i['target']] for i in edges]
            new_edges = [[node_map[u], node_map[v]] for u,v in edge_list]

            for i, j in zip(nodes, new_ids):
                i['id'] = j

            neuron_id = get_neuron_id(json_file)

            for node in nodes:
                if node['position'][2] == 0:
                    graph.add_nodes_from([
                        (node['id'], {
                            'x': convert_to_world(node['position'][0], voxel_size[0]),
                            'y': convert_to_world(node['position'][1], voxel_size[1]),
                            'z': convert_to_world(node['position'][2], voxel_size[2]),
                            'neuron_id': neuron_id
                            }
                    )
                ])
                else:
                    graph.add_nodes_from([
                        (node['id'], {
                            'x': convert_to_world(node['position'][0], voxel_size[0]),
                            'y': convert_to_world(node['position'][1], voxel_size[1]),
                            'z': convert_to_world((node['position'][2] - 1), voxel_size[2]),
                            'neuron_id': neuron_id
                            }
                    )
                ])


            for u,v in new_edges:
                graph.add_edge(u,v)

            start += length

    graph.write_nodes(roi)
    graph.write_edges(roi)


