import glob
import json
import numpy as np
import os
import sys


def get_neuron_id(in_file):

    in_file = os.path.basename(in_file)

    neuron_id = in_file.strip('.json')

    return int(neuron_id)

def convert_to_world(location, voxel_size):

    return [i*j for i,j in zip(location, voxel_size)]

if __name__ == '__main__':

    files = glob.glob(os.path.join(sys.argv[1], '*.json'))

    voxel_size = [10, 10, 20]

    start = 0

    node_positions = []
    total_edges = []

    for j in files:

        with open(j, 'r') as f:

            skel = json.load(f)

            nodes = skel['nodes']
            edges = skel['links']

            count = len(nodes)
            new_ids = list(range(start, start + count))

            neuron_id = get_neuron_id(j)

            node_map = {i['id']: j for i,j in zip(nodes, new_ids)}
            edge_list = [[i['source'], i['target']] for i in edges]
            new_edges = [[node_map[u], node_map[v]] for u,v in edge_list]

            for i,j in zip(nodes, new_ids):
                i['id'] = j

                node_positions.append([neuron_id, i['id'], convert_to_world(i['position'], voxel_size)])

            for u,v in new_edges:
                total_edges.append([u,v])

        start += count

    node_position_dict = {
            i[1]: np.array(i[2], dtype=np.float32)
            for i in node_positions
        }

    skeleton_lengths = {}

    for i,j in zip(node_positions, total_edges):
        skeleton_id = i[0]

        if skeleton_id not in skeleton_lengths:
            skeleton_lengths[skeleton_id] = 0

        pos_u = node_position_dict[j[0]]
        pos_v = node_position_dict[j[1]]

        # print(pos_u, pos_v)

        length = np.linalg.norm(pos_u - pos_v)

        skeleton_lengths[skeleton_id] += length

        total_length = np.sum([l for _, l in skeleton_lengths.items()])

        print(total_length)


