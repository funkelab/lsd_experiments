import pymongo
import networkx
import numpy as np
import daisy
import sys
import logging

# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'vanillaeuclidean_3sample_noglia_noautocontext'
# db_name = 'neuroglancer_lut_tests'
edges_collection = 'edges_hist_quant_50'
fragment_segment_lut_file = sys.argv[1]

def read_graph(seg_id):
    print('Loading lookup table...')
    fragment_segment_lut = np.load(
        fragment_segment_lut_file)['fragment_segment_lut']

    print('Filtering for fragments by segment id...')
    frag_ids = fragment_segment_lut[0]
    seg_ids = fragment_segment_lut[1]
    seg_fragments = frag_ids[seg_ids==seg_id]

    nodes = [
        {'id': f}
        for f in seg_fragments
    ]

    print('Reading graph for %d nodes...' % len(nodes))
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        edges_collection=edges_collection,
        mode='r',
        position_attribute=['center_z', 'center_y', 'center_x'])

    roi = daisy.Roi(
            (158000, 121800, 448616),
            (76000, 52000, 18944))

    edges = graph_provider.read_edges(roi=roi, nodes=nodes)
    edge_list = [
        (e['u'], e['v'], {'merge_score': e['merge_score']})
        for e in edges]

    print('Creating networkX graph...')
    graph = networkx.Graph()
    graph.add_nodes_from(seg_fragments)
    graph.add_edges_from(edge_list)

    return graph

def find_path(graph, frag_1, frag_2):

    print('Finding MST...')
    mst = networkx.minimum_spanning_tree(graph, weight='merge_score')
    path = networkx.shortest_path(mst, source=frag_1, target=frag_2)
    path = [ int(n) for n in path ]

    print('Getting node centers...')
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    nodes_collection = db['nodes']

    nodes = nodes_collection.find({'id': {'$in': path}}, {'_id': False})
    node_centers = {
        n['id']: np.array([n[x] for x in ['center_z', 'center_y', 'center_x']])
        for n in nodes
    }

    print(len(node_centers))

    print('Storing centers in array...')
    centers = np.array([
        [node_centers[u], node_centers[v]]
        for u, v in zip(path, path[1::])
    ])
    merge_scores = np.array([
        graph.edges[(u, v)]['merge_score']
        for u, v in zip(path, path[1::])
    ])

    print(centers[:10])
    print(merge_scores[:10])
    print(path)
    np.savez_compressed('%s_%s.npz' %(frag_1, frag_2), centers=centers, merge_scores=merge_scores)


if __name__ == '__main__':

    seg_id = 3469982
    frag_1 = 1001062924365 
    frag_2 = 1001062924381

    graph = read_graph(seg_id)
    find_path(graph, frag_1, frag_2)
