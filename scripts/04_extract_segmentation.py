import json
import logging
import lsd
import os
import daisy
import sys

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.mongodb_rag_provider').setLevel(logging.DEBUG)

def extract_segmentation(
        fragments_file,
        fragments_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        threshold,
        roi_offset=None,
        roi_shape=None):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r')

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    # slice
    print("Reading fragments and RAG in %s"%total_roi)
    fragments = fragments[total_roi]
    rag = rag_provider[total_roi]

    print("Number of nodes in RAG: %d"%(len(rag.nodes())))
    print("Number of edges in RAG: %d"%(len(rag.edges())))

    # create a segmentation
    print("Merging...")
    segmentation_data = fragments.to_ndarray()
    rag.get_segmentation(threshold, segmentation_data)

    # store segmentation
    print("Writing segmentation...")
    segmentation = daisy.prepare_ds(
        out_file,
        out_dataset,
        fragments.roi,
        fragments.voxel_size,
        fragments.data.dtype,
        # temporary fix until
        # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        # (we want gzip to be the default)
        compressor={'id': 'zlib', 'level':5})
    segmentation.data[:] = segmentation_data

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
