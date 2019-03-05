import daisy

mask_filename = '/groups/futusa/futusa/projects/fafb/calyx_neuropil_mask/calyx.zarr'
mask_ds = 'volumes/neuropil_mask'

db_host = 'slowpoke1'
db_name = 'synful'

nodes_collection = 'calyx_catmaid.nodes'
edges_collection = 'calyx_catmaid.edges'

roi = daisy.Roi(
    (158000, 121800, 403560),
    (76000, 52000, 64000))

if __name__ == "__main__":

    print("Opening graph db...")
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        host=db_host,
        db_name=db_name,
        nodes_collection=nodes_collection,
        edges_collection=edges_collection,
        position_attribute=['z', 'y', 'x'],
        node_attribute_collections={'calyx_neuropil_mask': ['masked']})

    graph = graph_provider[roi]

    print("Opening mask...")
    mask = daisy.open_ds(mask_filename, mask_ds)

    print("Masking skeleton nodes...")

    i = 0
    for node, data in graph.nodes(data=True):

        pos = daisy.Coordinate((data[d] for d in ['z', 'y', 'x']))
        data['masked'] = mask[pos]

        if i%1000 == 0:
            print("Masked %d/%d" % (i, graph.number_of_nodes()))

        i += 1

    graph.write_nodes()
