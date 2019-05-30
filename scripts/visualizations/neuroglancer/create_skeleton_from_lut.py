import pymongo
import networkx
import numpy as np
import daisy
import sys
import logging

# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'mtlsd_6sample_noglia_noautocontext'
# db_name = 'neuroglancer_lut_tests'
edges_collection = 'edges_hist_quant_50'
fragment_segment_lut_file ='/nrs/funke/sheridana/paper_runs/mtlsd/6_sample/no_glia/no_autocontext/setup101_p/400000/calyx.zarr/luts/fragment_segment/seg_edges_hist_quant_50_50.npz'

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
            (158000, 121800, 403560),
            (76000, 52000, 64000))

    edges = graph_provider.read_edges(roi=roi, nodes=nodes)
    edge_list = [
        (e['u'], e['v'], {'merge_score': e['merge_score']})
        for e in edges]

    print('Creating networkX graph...')
    graph = networkx.Graph()
    graph.add_nodes_from(seg_fragments)
    graph.add_edges_from(edge_list)

    graph = [ int(n) for n in graph ]

    print('Succesfully created networkX graph')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    nodes_collection = db['nodes']

    nodes = nodes_collection.find({'id': {'$in': graph}}, {'_id': False})
    node_centers = {
            n['id']: np.array([n[x] for x in ['center_z', 'center_y','center_x']]) 
            for n in nodes
    }

    centers = np.array([
        [node_centers[u], node_centers[v]]
        for u, v in zip(graph, graph[1::])
        ])

    np.savez_compressed('test.npz', centers=centers)

if __name__ == '__main__':

    seg_id = 73332973

    read_graph(seg_id)
