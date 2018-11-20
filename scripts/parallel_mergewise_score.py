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
from parallel_contingencies_map import parallel_contingencies

logging.basicConfig(level=logging.DEBUG)

def _merge_columns(counter, columns, new_column):
    """Returns sum of columns of ``counter`` specified in ``columns``."""
    merged = Counter()
    for key in counter.keys():
        if isinstance(key, tuple) and key[1] in columns:
            (gt_column, old_column) = key
            merged[(gt_column, new_column)] += counter[key]
        elif key in columns:
            merged[key] += counter[key]
    return merged

def _removed_columns(counter, columns):
    """Returns only columns of ``counter`` specified in ``columns``."""
    removed = Counter()
    for key in counter.keys():
        if isinstance(key, tuple) and key[1] in columns:
            removed[key] += counter[key]
        elif key in columns:
            removed[key] += counter[key]
    return removed

def _delta_entropy_col(counter, columns, total):
    """
    Returns change in entropy resulting from a merge of columns of ``counter``
    specified in ``columns``.
    """
    removed_columns = _removed_columns(counter, columns)
    merged_column = _merge_columns(counter, columns)
    entropy_to_remove = entropy_in_chunk(removed_columns, total)
    entropy_to_add = entropy_in_chunk(merged_column, total)
    return entropy_to_add - entropy_to_remove

def delta_entropy(contingencies,
                  fragment_counts,
                  components,
                  total,
                  num_workers):
    """
    Returns the calculated updates required to the entropies of the given
    ``contingencies`` and ``fragment_counts`` as a result of merging the
    fragments of each component in ``components``. This computation is
    performed in parallel with ``num_workers`` processes.
    """
    logging.info("Calculating entropy update for {0} merged components".format(len(components)))
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
    """
    Returns entropy contribution of ``chunk`` of larger array given the
    ``total`` value.
    """
    # assumes chunk is nonzero
    probabilities = chunk / total
    return np.sum(-probabilities * np.log2(probabilities))

def create_chunk_slices(total_size, chunk_size):
    """Returns slices of size min(remaining elements, ``chunk_size``)."""
    logging.debug("Creating chunks of size {0} for {1} elements".format(chunk_size, total_size))
    return [slice(i, min(i+chunk_size, total_size)) for i in range(0, total_size, chunk_size)]

def parallel_mergewise_score(rag,
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
    
    logging.info("Calculating entropies for fragments")

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
    logging.info("Fragment VOI split: {0} VOI merge: {1}".format(voi_split, voi_merge))
    results = [(voi_split, voi_merge)]

    for threshold in thresholds:
        logging.info("Calculating connected components for threshold {0}".format(threshold))
        components = rag.get_connected_components(threshold)
        (delta_H_contingencies, delta_H_seg) = delta_entropy(contingencies,
                                                             fragment_counts,
                                                             components,
                                                             total,
                                                             num_workers)
        voi_split = (H_contingencies + delta_H_contingencies) - H_gt_seg
        voi_merge = (H_contingencies + delta_H_contingencies) - (H_seg + delta_H_seg)
        results.append((voi_split, voi_merge))
