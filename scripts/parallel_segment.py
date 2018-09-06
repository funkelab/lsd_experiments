from pymongo import MongoClient
import daisy
import json
import logging
import lsd
import malis
import numpy as np
import os
import scipy
import sys
import waterz

logging.basicConfig(level=logging.INFO)

def parallel_segment(
        fragments,
        rag_provider,
        threshold,
        block_size,
        segmentation_ds):
    # prepare block ROIs
    total_roi = fragments.roi
    read_roi = daisy.Roi((0,)*fragments.roi.dims(), block_size)
    write_roi = daisy.Roi((0,)*fragments.roi.dims(), block_size)

    # find connected components of blocks in parallel
    logging.info("Starting blockwise connected component tasks")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: connected_components_in_block(
                rag_provider,
                threshold,
                b,
                segmentation_ds),
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel connected components failed, retrying %d/%d", i + 1, retry)
    
    # merge inter-block connected components in parallel
    # TODO


def connected_components_in_block(rag_provider, threshold, block, segmentation_ds):
    rag = rag_provider[block.read_roi]
    connected_components = rag.get_connected_components(threshold)
    for i, component in enumerate(connected_components):
        segmentation_ds[component.roi] = i
