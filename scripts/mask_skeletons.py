from funlib.segment.graphs import find_connected_components
import daisy
import time
import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)

mask_filename = '/groups/futusa/futusa/projects/fafb/calyx_neuropil_mask/calyx.zarr'
mask_ds = 'volumes/neuropil_mask'

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'synful'

nodes_collection = 'calyx_catmaid.nodes'
edges_collection = 'calyx_catmaid.edges'

roi = daisy.Roi(
    (158000, 121800, 403560),
    (76000, 52000, 44800))

# validation_roi = daisy.Roi(
    # (158000, 121800, 448360),
    # (76000, 52000, 19200))

# test_roi = daisy.Roi(
    # (158000, 121800, 403560),
    # (76000, 52000, 44800))

# calyx_roi = daisy.Roi(
    # (158000, 121800, 403560),
    # (76000, 52000, 64000))

if __name__ == "__main__":

    print("Opening graph db...")
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        nodes_collection=nodes_collection,
        edges_collection=edges_collection,
        endpoint_names=['source', 'target'],
        position_attribute=['z', 'y', 'x'],
        node_attribute_collections={
            'calyx_neuropil_mask_testing': ['masked'],
            'calyx_neuropil_components_testing': ['component_id'],
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
