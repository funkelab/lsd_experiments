import daisy
import os
import json
import logging
from lsd.local_segmentation import LocalSegmentationExtractor as LSE
import sys
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def segment_in_block(
        block,
        db_host,
        db_name,
        edges_collection,
        fragments_file,
        fragments_dataset,
        threshold,
        segmentation):

    logging.info('Creating local seg in block %s' %block.read_roi)
    start = time.time()

    local_seg = LSE(
                    db_host,
                    db_name,
                    edges_collection,
                    fragments_file,
                    fragments_dataset)

    local_seg = local_seg.get_local_segmentation(
            roi=block.write_roi,
            threshold=threshold).to_ndarray()

    logging.info("%.3fs"%(time.time() - start))

    segmentation[block.write_roi] = local_seg

def extract_local_segmentation(
        db_host,
        db_name,
        fragments_file,
        fragments_dataset,
        edges_collection,
        threshold,
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


    read_roi = daisy.Roi((0,)*3, (13950, 9300, 9300))
    write_roi = read_roi

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
            edges_collection,
            fragments_file,
            fragments_dataset,
            threshold,
            segmentation),
        fit='shrink',
        num_workers=num_workers,
        processes=True,
        read_write_conflict=False)

if __name__ == "__main__":

    db_host = "mongodb://lsdAdmin:C20H25N3O@funke-mongodb3.int.janelia.org:27017/admin?replicaSet=rsLsd"
    db_name = "hausser_test_neuron_setup19_230k_masked_mito_ff_10_no_eps"
    fragments_file = "/nrs/funke/sheridana/hausser/setup19/230000/neuron_test.zarr"
    fragments_dataset = "/volumes/fragments_masked_mito_ff_10_no_eps/s0"
    edges_collection = "edges_hist_quant_75"
    threshold = 0.4
    out_file = "/nrs/funke/sheridana/hausser/setup19/230000/neuron_test.zarr"
    out_dataset = "volumes/local_segmentation_40"
    num_workers = 20

    extract_local_segmentation(
        db_host,
        db_name,
        fragments_file,
        fragments_dataset,
        edges_collection,
        threshold,
        out_file,
        out_dataset,
        num_workers)



