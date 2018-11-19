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

def _merge_columns(csc_matrix, columns):
    """Returns sum of columns of ``csc_matrix`` specified in ``columns``."""
    return np.array(csc_matrix.[:,columns].sum(1))

def _removed_columns(csc_matrix, columns):
    """
    Returns all columns of ``csc_matrix`` specified in ``columns`` except
    first.
    """
    return csc_matrix[:,columns[1:]].toarray()

def _delta_entropy_col(csc_matrix, columns, total):
    """
    Returns change in entropy resulting from a merge of columns of ``csc_matrix
    specified in ``columns``.
    """
    removed_columns = _removed_columns(csc_matrix, columns)
    removed_elements = removed_columns[np.nonzero(removed_columns)]
    merged_column = _merge_columns(csc_matrix, columns)
    merged_elements = merged_column[np.nonzero(merged_column)]
    entropy_to_remove = entropy_in_chunk(removed_elements, total)
    entropy_to_add = entropy_in_chunk(merged_column)
    return entropy_to_add - entropy_to_remove

def delta_entropy(contingencies,
                  fragment_counts,
                  components,
                  total,
                  num_workers):
    delayed_delta_H_contingencies = [dask.delayed(_delta_entropy_col)(contingencies,
                                                                      c,
                                                                      total)
                                     for c in components]
    delayed_delta_H_seg = [dask.delayed(_delta_entropy_col)(contingencies,
                                                            c,
                                                            total)
                                     for c in components]
    delta_H_contingincies = dask.delayed(sum)(delayed_delta_H_contingencies).compute(num_workers=num_workers)
    delta_H_seg = dask.delayed(sum)(delayed_delta_H_contingencies).compute(num_workers=num_workers)
    return (delta_H_contingencies, delta_H_seg)

def entropy_in_chunk(chunk, total):
    # assumes chunk is nonzero
    probabilities = chunk / total
    return np.sum(-probabilities * np.log2(probabilities))

def create_chunk_slices(total_size, chunk_size):
    logging.debug("Creating chunks of size {0} for {1} elements".format(chunk_size, total_size))
    return [slice(i, min(i+chunk_size, total_size)) for i in range(0, total_size, chunk_size)]

def parallel_mergewise_score(
        rag,
        fragments_file,
        fragments_dataset,
        gt_seg_file,
        gt_seg_dataset,
        total_roi,
        block_size,
        chunk_size,
        thresholds,
        num_workers,
        retry):
    (contingencies,
     fragment_counts,
     gt_seg_counts,
     total) = parallel_contingencies(fragments_file,
                                     fragments_dataset,
                                     gt_seg_file,
                                     gt_seg_dataset,
                                     total_roi,
                                     block_size,
                                     num_workers,
                                     retry)
    
    seg = daisy.open_ds(seg_file, seg_dataset)
    gt_seg = daisy.open_ds(gt_seg_file, gt_seg_dataset)
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
                blocked_totals,
                seg_counts_shape,
                gt_seg_counts_shape,
                contingencies_shape),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel relabel failed, retrying %d/%d", i + 1, retry)

    logging.debug("Consolidating sparse information")

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
    
    logging.info("Calculating entropies")

    dask.config.set(scheduler='processes')
    contingencies_chunks = create_chunk_slices(contingencies.nnz, chunk_size)
    seg_counts_chunks = create_chunk_slices(seg_counts.nnz, chunk_size)
    gt_seg_counts_chunks = create_chunk_slices(gt_seg_counts.nnz, chunk_size)

    delayed_H_contingencies = [
            dask.delayed(entropy_in_chunk)(
                contingencies.data[c],
                total) for c in contingencies_chunks]
    delayed_H_seg = [
            dask.delayed(entropy_in_chunk)(
                seg_counts.data[c],
                total) for c in seg_counts_chunks]
    delayed_H_gt_seg = [
            dask.delayed(entropy_in_chunk)(
                gt_seg_counts.data[c],
                total) for c in gt_seg_counts_chunks]

    H_contingencies = dask.delayed(sum)(delayed_H_contingencies).compute(num_workers=num_workers)
    H_seg = dask.delayed(sum)(delayed_H_seg).compute(num_workers=num_workers)
    H_gt_seg = dask.delayed(sum)(delayed_H_gt_seg).compute(num_workers=num_workers)
    voi_split = H_contingencies - H_gt_seg
    voi_merge = H_contingencies - H_seg
    logging.info("VOI split: {0} VOI merge: {1}".format(voi_split, voi_merge))
    return (voi_split, voi_merge)
