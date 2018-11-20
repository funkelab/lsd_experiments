import dask
import dask.multiprocessing
import multiprocessing as mp
import daisy
import os
import logging
import shutil
import numpy as np
from scipy import sparse
from sys import argv, exit
from collections import Counter

logging.basicConfig(level=logging.DEBUG)

# allow for largest possible contingency table in sparse matrix
SEG_COUNTS_SHAPE = (int(10e7), 1)
GT_SEG_COUNTS_SHAPE = (1, int(10e7))
CONTINGENCIES_SHAPE = (int(10e7), int(10e7))

def contingencies_in_block(
        block,
        seg,
        gt_seg,
        contingencies,
        seg_counts,
        gt_seg_counts,
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
    logging.info("Calculating contingencies in {0}".format(block.read_roi))
    block_id = block.block_id
    seg_in_block =  seg[block.read_roi].to_ndarray()
    gt_seg_in_block = gt_seg[block.read_roi].to_ndarray()
    seg_indices = np.ravel(seg_in_block)
    gt_seg_indices = np.ravel(gt_seg_in_block)
    
    # ensure specified background values are ignored
    not_ignored = np.isin(gt_seg_indices, ignore, invert=True)
    seg_indices = seg_indices[not_ignored]
    gt_seg_indices = gt_seg_indices[not_ignored]
    
    # construct maps of partial counts
    partial_contingencies = Counter(np.stack([seg_indices, gt_seg_indices], axis=1).tolist())
    partial_seg_counts = Counter(seg_indices)
    partial_gt_seg_counts = Counter(gt_seg_indices)

    # append partial counts to shared memory lists
    contingencies.append(partial_contingencies)
    seg_counts.append(partial_seg_counts)
    gt_seg_counts.append(partial_gt_seg_counts)
    totals.append(len(gt_seg_indices))

def parallel_contingencies(seg_file,
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
    blocked_gt_seg_counts = m.list()
    blocked_totals = m.list()

    logging.info("Calculating contingencies")

    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: contingencies_in_block(
                b,
                seg,
                gt_seg,
                blocked_contingencies,
                blocked_seg_counts,
                blocked_gt_seg_counts,
                blocked_totals),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel contingencies failed, retrying %d/%d", i + 1, retry)

    logging.debug("Consolidating sparse partial counts")

    total = np.sum(np.uint64(blocked_totals))
    contingencies = Counter()
    seg_counts = Counter()
    gt_seg_counts = Counter()

    for block in blocked_contingencies:
        contingencies += block
    for block in blocked_seg_counts:
        seg_counts += block
    for block in blocked_gt_seg_counts:
        gt_seg_counts += block

    return (contingencies, seg_counts, gt_seg_counts, total)
