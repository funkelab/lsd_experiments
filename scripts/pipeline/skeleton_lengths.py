import daisy
import numpy as np

db_host ='mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke' 
old_db_name = 'zebrafinch_gt_skeletons_test_infinite_roi'
new_db_name = 'zebrafinch_gt_skeletons_test_infinite_roi_new_gt_2'

roi = daisy.Roi((-10e6,)*3, (20e6,)*3)

# roi = daisy.Roi((10000,)*3, (5000,)*3)

def get_skeletons_in_roi(roi, db_name):

    print('Opening graph...')

    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection='zebrafinch.nodes',
            edges_collection='zebrafinch.edges',
            mode='r',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'])

    skeletons = skeletons_provider.get_graph(roi)

    return skeletons

def get_skeleton_lengths(skeletons):

    node_positions = {
            node: np.array(
                [
                    skeletons.nodes[node][d]
                    for d in ['z', 'y', 'x']
                ],
                dtype=np.float32)
            for node in skeletons.nodes()
        }

    skeleton_lengths = {}

    for u, v, data in skeletons.edges(data=True):

        skeleton_id = skeletons.nodes[u]['neuron_id']

        if skeleton_id not in skeleton_lengths:
            skeleton_lengths[skeleton_id] = 0

        pos_u = node_positions[u]
        pos_v = node_positions[v]

        length = np.linalg.norm(pos_u - pos_v)

        skeleton_lengths[skeleton_id] += length

        total_length = np.sum([l for _, l in skeleton_lengths.items()])

    return total_length

if __name__ == '__main__':

    old_skeletons = get_skeletons_in_roi(roi, old_db_name)
    new_skeletons = get_skeletons_in_roi(roi, new_db_name)

    old_total_length = get_skeleton_lengths(old_skeletons)
    new_total_length = get_skeleton_lengths(new_skeletons)

    print(old_total_length, new_total_length)
