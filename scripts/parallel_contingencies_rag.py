import dask
import dask.multiprocessing
import multiprocessing as mp
import lsd
import daisy
import os
import logging
import shutil
import numpy as np
from scipy import sparse
from sys import argv, exit
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def relabel_blocked_volume(volume, fragments_map, ignore=[0]):
    logger.info("Relabeling volume")
    fragments = np.unique(volume)
    fragments = fragments[np.isin(fragments, ignore, invert=True)]

    # shift fragment values to potentially save memory when relabeling
    if len(fragments) > 0:
        min_fragment = fragments.min()
        offset = 0
        if min_fragment > 0:
            offset = fragments.dtype.type(min_fragment - 1)
        volume[np.isin(volume, ignore, invert=True)] -= offset
        shifted_fragments = fragments - offset
        
        components = []
        for fragment in fragments:
            if fragment in fragments_map:
                components.append(fragments_map[fragment])
        components = np.array(components, dtype=fragments.dtype)
        logger.debug("Restricted fragments map to {0} elements".format(len(components)))
        volume = lsd.labels.replace_values(volume, shifted_fragments, components)
        logger.debug("Finished relabeling {0} fragments".format(len(components)))
        return volume
    else:
        logging.debug("Block {0} contains no relevant fragments".format(block.block_id))
        return volume

def contingencies_in_block(
        block,
        seg,
        gt_seg,
        components,
        fragments_map,
        contingencies,
        seg_counts,
        totals,
        ignore=[0]):
    """
    Calculates contingencies in block. The table has structure:

                  pred seg   ...
             +__________________
             |_|_|_|_|_|_|_|_|_
             |_|_|_|_|_|_|_|_|_
             |_|_|_|_|_|_|_|_|_
    gt seg   |_|_|_|_|_|_|_|_|_
             |_|_|_|_|_|_|_|_|_
           . |_|_|_|_|_|_|_|_|_
           . |_|_|_|_|_|_|_|_|_
           . |

    """
    # read data in block
    logger.info("Calculating contingencies in {0}".format(block.read_roi))
    block_id = block.block_id
    seg_in_block =  seg[block.read_roi].to_ndarray()
    seg_in_block = relabel_blocked_volume(seg_in_block, fragments_map, ignore)
    gt_seg_in_block = gt_seg[block.read_roi].to_ndarray()
    seg_indices = np.ravel(seg_in_block)
    gt_seg_indices = np.ravel(gt_seg_in_block)
    
    # ensure specified background values are ignored
    not_ignored = np.isin(gt_seg_indices, ignore, invert=True)
    seg_indices = seg_indices[not_ignored]
    gt_seg_indices = gt_seg_indices[not_ignored]
    
    # construct maps of partial counts
    partial_contingencies = Counter([tuple(l) for l in np.stack([gt_seg_indices, seg_indices], axis=1).tolist()])
    partial_seg_counts = Counter(seg_indices.tolist())

    # append partial counts to shared memory lists
    contingencies.append(partial_contingencies)
    seg_counts.append(partial_seg_counts)
    totals.append(len(gt_seg_indices))

def parallel_contingencies_rag(components,
                               seg_file,
                               seg_dataset,
                               gt_seg_file,
                               gt_seg_dataset,
                               total_roi,
                               block_size,
                               num_workers,
                               retry):
    """
    Constructs contingency table between a predicted segmentation and a ground
    truth segmentation blockwise in parallel. Returns sparse (CSC) matrices
    for the contingencies, the predicted segmentation ID counts, the ground
    truth segmentation ID counts, and the total value in the contingency table.

    Also see ``contingencies_in_block`` for details on contingency table
    structure.

    Args:

        seg_file (``string``):

            The input h5-like file containing predicted segmentation.

        seg_dataset (``string``):

            Name of dataset containing predicted segmentation.

        gt_seg_file (``string``):

            The h5-like file containing ground truth segmentation.

        gt_seg_dataset (``string``):

            Name of dataset containing ground truth segmentation.

        total_roi (``daisy.Roi``):

            The desired ROI in which to calculate contingencies.

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        num_workers (``int``):

            How many blocks to run in parallel.

        retry (``int``):

            Number of repeat attempts if any tasks fail in first run.
    """
    # define block ROIs for parallel processing
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    
    # load segmentation and ground truth segmentation
    seg = daisy.open_ds(seg_file, seg_dataset)
    gt_seg = daisy.open_ds(gt_seg_file, gt_seg_dataset)

    # construct shared memory lists to store partial counts from blocks
    m = mp.Manager()
    blocked_contingencies = m.list()
    blocked_seg_counts = m.list()
    blocked_totals = m.list()
    
    logging.info("Constructing dictionary from fragments to components")
    fragments_map = {fragment: i+1 for i, component in enumerate(components) for fragment in component}
    print(len(fragments_map))

    logger.info("Calculating contingencies")

    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: contingencies_in_block(
                b,
                seg,
                gt_seg,
                components,
                fragments_map,
                blocked_contingencies,
                blocked_seg_counts,
                blocked_totals),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logger.error("parallel contingencies failed, retrying %d/%d", i + 1, retry)

    logger.debug("Consolidating sparse partial counts")

    total = np.sum(np.uint64(blocked_totals))
    contingencies = Counter()
    seg_counts = Counter()

    for block in blocked_contingencies:
        contingencies += block
    for block in blocked_seg_counts:
        seg_counts += block

    return (contingencies, seg_counts, total)
