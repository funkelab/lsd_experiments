from pymongo import MongoClient
import daisy
import json
import logging
import lsd
import sys
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def segment_in_block(
        block,
        db_host,
        db_name,
        fragment_segment_collection,
        segmentation,
        fragments):

    logging.info("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.write_roi)
    fragment_ids = np.unique(fragments)
    logging.info("%.3fs"%(time.time() - start))

    logging.info("Found %d fragments"%len(fragment_ids))

    # get segments

    client = MongoClient(db_host)
    database = client[db_name]
    collection = database[fragment_segment_collection]

    logging.info("Reading fragment-segment LUT...")
    start = time.time()
    fragments_map = collection.find(
        {
            'fragment': { '$in': list([ int(f) for f in fragment_ids]) },
        })
    logging.info("%.3fs"%(time.time() - start))

    fragments_map = list(fragments_map)
    logging.info("Found segments for %d fragments"%len(fragments_map))

    logging.info("Building fragments map...")
    fragments_map = {
        f['fragment']: f['segment']
        for f in fragments_map
    }

    segment_ids = np.array([
        fragments_map.get(fragment, fragment)
        for fragment in fragment_ids
    ], dtype=fragments.dtype)
    logging.info("%.3fs"%(time.time() - start))

    # shift fragment values to potentially save memory when relabeling
    min_fragment = fragment_ids.min()
    offset = 0
    if min_fragment > 0:
        offset = fragment_ids.dtype.type(min_fragment - 1)
        fragments -= offset
        fragment_ids -= offset

    logging.info("Mapping fragments to %d segments", len(segment_ids))
    start = time.time()
    relabelled = lsd.labels.replace_values(fragments, fragment_ids, segment_ids)
    logging.info("%.3fs"%(time.time() - start))

    segmentation[block.write_roi] = relabelled

def extract_segmentation(
        fragments_file,
        fragments_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        fragment_segment_collection,
        roi_offset=None,
        roi_shape=None,
        num_workers=1,
        **kwargs):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    write_roi = daisy.Roi((0, 0, 0), (1040, 1024, 1024))
    read_roi = daisy.Roi((0, 0, 0), (1040, 1024, 1024))
    # write_roi = daisy.Roi((0, 0, 0), (200, 256, 256))
    # read_roi = daisy.Roi((0, 0, 0), (200, 256, 256))

    logging.info("Preparing segmentation dataset...")
    segmentation = daisy.prepare_ds(
        out_file,
        out_dataset,
        total_roi,
        voxel_size=fragments.voxel_size,
        dtype=np.uint64,
        write_roi=write_roi)

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: segment_in_block(
            b,
            db_host,
            db_name,
            fragment_segment_collection,
            segmentation,
            fragments),
        fit='shrink',
        num_workers=num_workers,
        processes=True,
        read_write_conflict=False)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)

