import csv
import glob
import os
import sys
from xml.dom import minidom
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import BulkWriteError

from nml_parser import *

def get_neuron_id(in_file, prefix):

    in_file = os.path.basename(in_file)

    # neuron_id = in_file.strip(prefix).strip('.nml')
    neuron_id = in_file.strip('.nml')

    return int(neuron_id)

def convert_to_world(location, voxel_size):

    # nml starts with 1 and is in voxel space

    return (location * voxel_size)

if __name__ == '__main__':

    voxel_size = [9, 9, 20]

    files = glob.glob(os.path.join(sys.argv[1], '*.nml'))

    start = 0

    for nml_file in files:

        print('loading %s'%nml_file)

        nodes, edges = parse_nml(nml_file)

        length = len(nodes.keys())

        new_ids = list(range(start,start+length))

        #remap nodes start at zero
        node_map = {i:j for i,j in zip(nodes.keys(), new_ids)}
        new_nodes = {node_map[i]:j for i,j in nodes.items()}

        #remap edges
        new_edges = [[node_map[u],node_map[v]] for u,v in edges]

        neuron_id = get_neuron_id(nml_file, 'test_set_skeleton')

        with open('gt_skels/val_zebrafinch_nodes_%i.csv'%neuron_id, 'w') as f:
            writer=csv.writer(f)
            writer.writerow(
                    ["neuron id",
                     "node id",
                     "x",
                     "y",
                     "z"
                     ]
                )

            for node_id, node in new_nodes.items():

                print(node_id, int(node.position[0]), int(node.position[1]), int(node.position[2]))

                row = [
                        neuron_id,
                        node_id,
                        int(convert_to_world(node.position[0], voxel_size[0])),
                        int(convert_to_world(node.position[1], voxel_size[1])),
                        int(convert_to_world(node.position[2], voxel_size[2])),
                    ]

                writer.writerow(row)

        with open('gt_skels/val_zebrafinch_edges_%i.csv'%neuron_id, 'w') as f:
            writer=csv.writer(f)
            writer.writerow(["u","v"])

            for u,v in new_edges:
                print(neuron_id, u, v)

                row = [u, v]
                writer.writerow(row)

        start += length
