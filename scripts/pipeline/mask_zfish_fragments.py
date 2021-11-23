from funlib.segment.graphs import find_connected_components
import daisy
import time
import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)

mask_filename = '/nrs/funke/sheridana/zebrafish_nuclei/130201zf142.zarr'
mask_ds = 'volumes/labels/mask/s0'

db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'zebrafish_nuclei_auto_basic_70k_testing_mask_test'

nodes_collection = 'nodes'

# roi = daisy.Roi((187200, 62720, 173712), (20000, 20000, 20000))

if __name__ == "__main__":

    print("Opening mask...")
    mask = daisy.open_ds(mask_filename, mask_ds)

    roi = mask.roi

    print("Opening graph db...")
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        nodes_collection=nodes_collection,
        position_attribute=['z', 'y', 'x'],
        node_attribute_collections={
            'masked': ['masked']}
        )

    print("Reading graph in %s" % roi)
    start = time.time()
    graph = graph_provider[roi]
    print("%.3fs"%(time.time() - start))


    print("Masking nodes...")

    # print(graph.nodes)

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

    print("Original graph contains %d nodes" % graph.number_of_nodes())
    print("Filtered graph contains %d nodes" % filtered_graph.number_of_nodes())

    graph.write_nodes()
