from funlib.segment.graphs import find_connected_components
import daisy
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'zebrafinch_gt_skeletons_new_gt_9_9_20_testing'

nodes_collection = 'zebrafinch.nodes'
edges_collection = 'zebrafinch.edges'

if __name__ == "__main__":

    path_to_configs = sys.argv[1]

    for i in range(1,11):

        with open(os.path.join(path_to_configs, 'zebrafinch_11_micron_roi_%i.json'%i), 'r') as f:
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
                'zebrafinch_components_11_micron_roi_not_masked_not_relabelled_%i'%i: ['component_id'],
            })

        print("Reading graph in %s" % roi)
        start = time.time()
        graph = graph_provider[roi]
        print("%.3fs"%(time.time() - start))

        remove_nodes = []
        filtered_graph = daisy.Graph(graph_data=graph)
        # remove outside edges and nodes
        for node, data in filtered_graph.nodes(data=True):
            if 'z' not in data:
                remove_nodes.append(node)
        print("Removing %d nodes that were outside of ROI"%len(remove_nodes))
        for node in remove_nodes:
            filtered_graph.remove_node(node)

        print("Original graph contains %d nodes, %d edges" %
            (graph.number_of_nodes(), graph.number_of_edges()))
        print("Filtered graph contains %d nodes, %d edges" %
            (filtered_graph.number_of_nodes(), filtered_graph.number_of_edges()))

        # relabel connected components
      #   print("Relabeling skeleton components...")
        # start = time.time()
        # node_to_comp = find_connected_components(
            # filtered_graph,
            # 'component_id',
            # return_lut=True)
        # print("%.3fs"%(time.time() - start))

        for node, data in graph.nodes(data=True):
            if 'neuron_id' in data:
                data['component_id'] = data['neuron_id']
            else:
                data['component_id'] = -1

        graph.write_nodes()
