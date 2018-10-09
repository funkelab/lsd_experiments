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
import z5py

logging.basicConfig(level=logging.DEBUG)

def relabel_in_block(block, segmentation_ds, fragments_map, ignore=[0]):
    logging.debug("Relabeling in {0}".format(block.read_roi))
    volume = segmentation_ds[block.read_roi].data
    fragments = np.unique(volume)
    fragments = fragments[np.isin(fragments, ignore, invert=True)]

    # shift fragment values to potentially save memory when relabeling
    min_fragment = fragments.min()
    assert min_fragment > 0
    offset = fragments.dtype.type(min_fragment - 1)
    volume[np.isin(volume, ignore, invert=True)] -= offset
    shifted_fragments = fragments - offset

    components = [fragments_map[fragment] for fragment in fragments]
    logging.debug("Restricted fragments map to {0} elements".format(len(components)))
    volume = lsd.labels.replace_values(volume, shifted_fragments, components)
    logging.debug("Writing relabeled block to segmentation volume")
    segmentation_ds[block.read_roi] = volume

def parallel_relabel(
        components,
        fragments,
        total_roi,
        block_size,
        tmp_fname,
        tmp_dsname,
        num_workers,
        retry):
    
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    
    logging.info("Constructing temporary segmentation dataset")
    if os.path.isdir(os.path.join(tmp_fname, tmp_dsname)):
        shutil.rmtree(os.path.join(tmp_fname, tmp_dsname))
    shutil.copytree(fragments.data.path, os.path.join(tmp_fname, tmp_dsname))
    segmentation_ds = daisy.open_ds(tmp_fname, tmp_dsname)

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
                fragments_map),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel relabel failed, retrying %d/%d", i + 1, retry)
