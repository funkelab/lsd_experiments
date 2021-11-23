import daisy
import json
import logging
import os
import sys
import time
from funlib.segment.graphs import find_connected_components

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)

mask_filename = '/nrs/funke/sheridana/zebrafinch/zebrafinch_realigned.zarr'
mask_ds = 'volumes/ffn_neuropil_mask/s0'

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
# db_host = 'mongodb://lsdAdmin:C20H25N3O@funke-mongodb3.int.janelia.org:27017/admin?replicaSet=rsLsd'
db_name = 'zebrafinch_gt_skeletons_new_gt_9_9_20_validation'

nodes_collection = 'zebrafinch.nodes'
edges_collection = 'zebrafinch.edges'

# roi = daisy.Roi(
    # (21000, 13050, 12150),
    # (72000, 72000, 72000))

if __name__ == "__main__":

    # rois = ['11', '18', '25', '32', '40', '47', '54', '61', '68', '76']

    rois = ['76']

    # rois = ['benchmark']

    path_to_configs = sys.argv[1]

    for i in rois:

        with open(os.path.join(path_to_configs, 'zebrafinch_%s_micron_roi.json'%i), 'r') as f:
            config = json.load(f)

        roi = daisy.Roi(
                tuple(config['roi_offset']),
                tuple(config['roi_shape']))

        print(roi)

        print("Opening graph db...")
        graph_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'],
            node_attribute_collections={
                'zebrafinch_mask_%s_micron_roi_masked'%i: ['masked'],
                'zebrafinch_components_%s_micron_roi_masked'%i: ['component_id'],
            })

        print("Reading graph in %s" % roi)
        start = time.time()
        graph = graph_provider[roi]
        print("%.3fs"%(time.time() - start))

        print("Opening mask...")
        mask = daisy.open_ds(mask_filename, mask_ds)

        print("Masking skeleton nodes...")

        i = 0
        start = time.time()
        for node, data in graph.nodes(data=True):

            if 'z' not in data:
                continue

            pos = daisy.Coordinate((data[d] for d in ['z', 'y', 'x']))
            data['masked'] = bool(mask[pos])

            if i%1000 == 0:
                print("Masked %d/%d" % (i, graph.number_of_nodes()))

            i += 1

        print("%.3fs"%(time.time() - start))

        remove_nodes = []
        filtered_graph = daisy.Graph(graph_data=graph)
        # remove outside edges and nodes
        for node, data in filtered_graph.nodes(data=True):
            if 'z' not in data or not data['masked']:
                remove_nodes.append(node)
        print("Removing %d nodes that were outside of ROI or not masked"%len(remove_nodes))
        for node in remove_nodes:
            filtered_graph.remove_node(node)

        print("Original graph contains %d nodes, %d edges" %
            (graph.number_of_nodes(), graph.number_of_edges()))
        print("Filtered graph contains %d nodes, %d edges" %
            (filtered_graph.number_of_nodes(), filtered_graph.number_of_edges()))

        # relabel connected components
        print("Relabeling skeleton components...")
        start = time.time()
        node_to_comp = find_connected_components(
            filtered_graph,
            'component_id',
            return_lut=True)
        print("%.3fs"%(time.time() - start))

        for node, data in graph.nodes(data=True):
            if node in node_to_comp:
                data['component_id'] = int(node_to_comp[node])
            else:
                data['component_id'] = -1

        graph.write_nodes()
