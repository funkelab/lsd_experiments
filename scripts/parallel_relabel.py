import dask
import daisy
import lsd
import os
import sys
import shutil
import logging
from concurrent.futures  import ProcessPoolExecutor as Pool
from multiprocessing import Process, sharedctypes
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def relabel_in_block(block, segmentation_ds, fragments_ds, fragments_map, ignore=[0]):
    logging.info("Relabeling in {0}".format(block.read_roi))
    volume = fragments_ds[block.read_roi].to_ndarray()
    fragments = np.unique(volume)
    # fragments = fragments[np.isin(fragments, ignore, invert=True)]

    # shift fragment values to potentially save memory when relabeling
    if len(fragments) > 0:
        min_fragment = fragments.min()
        offset = 0
        if min_fragment > 0:
            offset = fragments.dtype.type(min_fragment - 1)
        # volume[np.isin(volume, ignore, invert=True)] -= offset
        volume -= offset
        shifted_fragments = fragments - offset

        components = np.array([fragments_map[fragment] for fragment in fragments], dtype=fragments.dtype)
        logging.debug("Restricted fragments map to {0} elements".format(len(components)))
        volume = lsd.labels.replace_values(volume, shifted_fragments, components)
        logging.debug("Writing relabeled block to segmentation volume")
        segmentation_ds[block.read_roi] = volume
    else:
        logging.debug("Block {0} contains no relevant fragments".format(block.block_id))

def parallel_relabel(
        components,
        fragments_file,
        fragments_dataset,
        total_roi,
        block_size,
        seg_file,
        seg_dataset,
        num_workers,
        retry):
    
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    fragments_ds = daisy.open_ds(fragments_file, fragments_dataset)
    
    logging.info("Constructing temporary segmentation dataset")
    segmentation_ds = daisy.prepare_ds(seg_file, seg_dataset, total_roi, fragments_ds.voxel_size, dtype=fragments_ds.dtype)

    logging.info("Constructing dictionary from fragments to components")
    fragments_map = {fragment: i+1 for i, component in enumerate(components) for fragment in component}

    logging.info("Starting relabeling tasks for {0} components".format(len(components)))
    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: relabel_in_block(
                b,
                segmentation_ds,
                fragments_ds,
                fragments_map),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel relabel failed, retrying %d/%d", i + 1, retry)
