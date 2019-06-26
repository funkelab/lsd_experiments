import daisy
import json
import logging
import lsd
import sys
import time
from parallel_read_rag import parallel_read_rag
from parallel_relabel import parallel_relabel

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.mongodb_rag_provider').setLevel(logging.DEBUG)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def extract_segmentation(
        fragments_file,
        fragments_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        edges_collection,
        threshold,
        roi_offset=None,
        roi_shape=None,
        num_workers=1,
        **kwargs):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection)

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    # slice
    print("Reading fragments and RAG in %s"%total_roi)
    start = time.time()
    fragments = fragments[total_roi]
    rag = parallel_read_rag(
        total_roi,
        db_host,
        db_name,
        edges_collection=edges_collection,
        block_size=(4096, 4096, 4096),
        num_workers=num_workers,
        retry=0)
    print("%.3fs"%(time.time() - start))

    print("Number of nodes in RAG: %d"%(len(rag.nodes())))
    print("Number of edges in RAG: %d"%(len(rag.edges())))

    # create a segmentation
    print("Merging...")
    start = time.time()
    components = rag.get_connected_components(threshold)
    print("%.3fs"%(time.time() - start))

    print("Constructing dictionary from fragments to segments")
    fragments_map = {fragment: component[0] for component in components for fragment in component}

    print("Writing segmentation...")
    start = time.time()
    parallel_relabel(
        fragments_map,
        fragments_file,
        fragments_dataset,
        total_roi,
        block_size=(4080, 4096, 4096),
        seg_file=out_file,
        seg_dataset=out_dataset,
        num_workers=num_workers,
        retry=0)
    print("%.3fs"%(time.time() - start))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
