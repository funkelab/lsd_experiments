import daisy
import os
import json
import logging
from funlib.segment.arrays import replace_values
import sys
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def segment_in_block(
        block,
        fragments_file,
        fragment_segment_lut,
        segmentation,
        fragments):

    logging.info("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.write_roi)
    logging.info("%.3fs"%(time.time() - start))

    # get segments

    lut = os.path.join(fragments_file, fragment_segment_lut)
    assert os.path.exists(lut), "%s does not exist" % lut

    logging.info("Reading fragment-segment LUT...")
    lut = np.load(lut)
    logging.info("%.3fs"%(time.time() - start))

    logging.info("Found %d fragments in LUT"%len(lut[0]))

    num_segments = len(np.unique(lut[1]))
    logging.info("Relabelling fragments to %d segments", num_segments)
    start = time.time()
    relabelled = replace_values(fragments, lut[0], lut[1])
    logging.info("%.3fs"%(time.time() - start))

    segmentation[block.write_roi] = relabelled

def extract_segmentation(
        fragments_file,
        fragments_dataset,
        fragment_segment_lut,
        out_file,
        out_dataset,
        num_workers,
        roi_offset=None,
        roi_shape=None,
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
            fragments_file,
            fragment_segment_lut,
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

