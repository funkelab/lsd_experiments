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

logging.basicConfig(level=logging.DEBUG)

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
    Calculates contingencies in block. The table should have structure:

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
    num_indices = seg_indices.shape
    
    # allow for largest possible contingency table in sparse matrix
    seg_counts_shape = (1, int(10e7))
    gt_seg_counts_shape = (int(10e7), 1)
    contingencies_shape = (int(10e7), int(10e7))
    
    # ensure certain indices are ignored
    data = np.ones(seg_indices.shape)
    ignored = np.isin(gt_seg_indices, ignore)
    data[ignored] = 0
    
    # construct sparse matrices of partial counts
    partial_contingencies = sparse.coo_matrix(
            (data, (gt_seg_indices, seg_indices)),
            shape=contingencies_shape).tocsc()
    partial_seg_counts = sparse.coo_matrix(
            (data, (seg_indices, np.zeros(seg_indices.shape))),
            shape=seg_counts_shape).tocsc()
    partial_gt_seg_counts = sparse.coo_matrix(
            (data, (np.zeros(seg_indices.shape), gt_seg_indices)),
            shape=gt_seg_counts_shape).tocsc()

    # append partial counts to shared memory lists
    contingencies.append(partial_contingencies)
    seg_counts.append(partial_seg_counts)
    gt_seg_counts.append(partial_gt_seg_counts)
    totals.append(np.sum(data))

def parallel_contingencies(seg_file,
                           seg_dataset,
                           gt_seg_file,
                           gt_seg_dataset,
                           total_roi,
                           block_size,
                           chunk_size,
                           num_workers,
                           retry):
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

    total = np.float64(np.sum(blocked_totals))
    contingencies = sparse.csc_matrix(contingencies_shape, dtype=np.uint64)
    seg_counts = sparse.csc_matrix(seg_counts_shape, dtype=np.uint64)
    gt_seg_counts = sparse.csc_matrix(gt_seg_counts_shape, dtype=np.uint64)

    for block in blocked_contingencies:
        contingencies += block
    for block in blocked_seg_counts:
        seg_counts += block
    for block in blocked_gt_seg_counts:
        gt_seg_counts += block

    return (contingencies, seg_counts, gt_seg_counts, total)
