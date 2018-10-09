import dask
import daisy
import lsd
import os
import sys
import shutil
import logging
from concurrent.futures  import ProcessPoolExecutor as Pool
from multiprocessing import Process, Manager, sharedctypes
import numpy as np
import z5py

logging.basicConfig(level=logging.DEBUG)

def unique_in_block(block, segmentation_ds, id_registry, ignore=[0]):
    volume = segmentation_ds[block.read_roi].data
    unique = np.unique(volume)
    unique = unique[np.isin(unique, ignore, invert=True)]
    dict_entries = np.vstack((unique, np.ones(unique.shape)))
    dict_entries = dict_entries.T
    id_registry.update(dict_entries)

def renumber_in_block(block, segmentation_ds, id_map, ignore=[0]):
    logging.debug("Relabeling in {0}".format(block.read_roi))
    volume = segmentation_ds[block.read_roi].data
    old_ids = np.unique(volume)
    old_ids = old_ids[np.isin(old_ids, ignore, invert=True)]

    # shift ID values to potentially save memory when relabeling
    min_id = old_ids.min()
    assert min_id > 0
    offset = old_ids.dtype.type(min_id - 1)
    volume[np.isin(volume, ignore, invert=True)] -= offset
    shifted_old_ids = old_ids - offset

    new_ids = [id_map[old_id] for old_id in old_ids]
    logging.debug("Restricted ID map to {0} elements".format(len(new_ids)))
    volume = lsd.labels.replace_values(volume, shifted_old_ids, new_ids)
    logging.debug("Writing relabeled block to segmentation volume")
    segmentation_ds[block.read_roi] = volume

def parallel_renumber(
        segmentation_ds,
        total_roi,
        voxel_size,
        block_size,
        num_workers,
        retry):
    
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
 
    m = Manager()
    id_registry = m.dict()

    logging.info("Starting unique ID registration tasks")
    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: unique_in_block(
                b,
                segmentation_ds,
                id_registry),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel ID registration failed, retrying %d/%d", i + 1, retry)

    id_map = {old_id: i+1 for i, old_id in enumerate(id_registry.values())}

    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: renumber_in_block(
                b,
                segmentation_ds,
                id_map),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel renumbering failed, retrying %d/%d", i + 1, retry)

    return len(id_map)
