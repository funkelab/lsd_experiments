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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _merged_columns_2D(counter, new_column):
    """Implements ``_merge_columns`` for a matrix with >1 rows."""
    merged = Counter()
    for key in counter.keys():
        (gt_column, old_column) = key
        merged[(gt_column, new_column)] += counter[key]
    return merged

def _merged_columns_1D(counter, new_column):
    """Implements ``_merge_columns`` for a row vector."""
    merged = Counter()
    for key in counter.keys():
        merged[new_column] += counter[key]
    return merged

def _merged_columns(counter, new_column):
    """
    Creates new column with ID ``new_column`` containing sum of all columns of
    ``counter``.
    """
    keys = list(counter.keys())
    if len(keys) == 0:
        return Counter()
    if isinstance(keys[0], tuple):
        merged = _merged_columns_2D(counter, new_column)
    else:
        merged = _merged_columns_1D(counter, new_column)
    return merged

def _removed_columns_2D(counter, columns):
    """Implements ``_removed_columns`` for a matrix with >1 rows."""
    removed = Counter()
    for key in counter.keys():
        if key[1] in columns:
            removed[key] += counter[key]
    return removed

def _removed_columns_1D(counter, columns):
    """Implements ``_removed_columns`` for a row vector."""
    removed = Counter()
    for key in counter.keys():
        if key in columns:
            removed[key] += counter[key]
    return removed

def _removed_columns(counter, columns):
    """Returns only columns of ``counter`` specified in ``columns``."""
    if isinstance(list(counter.keys())[0], tuple):
        removed = _removed_columns_2D(counter, columns)
    else:
        removed = _removed_columns_1D(counter, columns)
    return removed

def _update_columns_2D(counter, columns, new_column):
    """Implements ``_update_columns`` for a matrix with >1 rows."""
    removed = Counter()
    merged = Counter()
    for key in counter.keys():
        if key[1] in columns:
            (gt_column, old_column) = key
            removed[key] += counter[key]
            merged[(gt_column, new_column)] += counter[key]
    return removed, merged

def _update_columns_1D(counter, columns, new_column):
    """Implements ``_update_columns`` for a row vector."""
    removed = Counter()
    merged = Counter()
    for key in counter.keys():
        if key in columns:
            removed[key] += counter[key]
            merged[new_column] += counter[key]
    return removed, merged

def _update_columns(counter, columns, new_column):
    """
    Returns removed columns and merged column of ``counter`` specified in
    ``columns``.
    """
    if isinstance(list(counter.keys())[0], tuple):
        removed, merged = _update_columns_2D(counter, columns, new_column)
    else:
        removed, merged = _update_columns_1D(counter, columns, new_column)

    return removed, merged

def _delta_entropy_col(counter, columns, total, new_column):
    """
    Returns change in entropy resulting from a merge of columns of ``counter``
    specified in ``columns``.
    """
    removed_columns_counter = _removed_columns(counter, columns)
    removed_columns = np.array(list(removed_columns_counter.values()),
                                dtype=np.float64)
    merged_column = np.array(list(_merged_columns(removed_columns_counter,
                                                  new_column).values()),
                             dtype=np.float64)
    removed_columns = removed_columns[np.nonzero(removed_columns)]
    merged_column = merged_column[np.nonzero(merged_column)]

    if removed_columns.size > 0:
        entropy_to_remove = entropy_in_chunk(removed_columns,
                                             total)
    else:
        entropy_to_remove = 0.0

    if merged_column.size > 0:
        entropy_to_add = entropy_in_chunk(merged_column,
                                          total)
    else:
        entropy_to_add = 0.0

    return entropy_to_add - entropy_to_remove

def delta_entropy_in_chunk(counter, components_in_chunk, chunk, num_components, total):
    logger.info("Calculating entropy update for {0} merged components".format(len(components_in_chunk)))
    delta = 0
    new_columns = range(*chunk.indices(num_components))
    for i, component in enumerate(components_in_chunk):
        delta += _delta_entropy_col(counter, component, total, new_columns[i])
    return delta

def delta_entropy(contingencies,
                  fragment_counts,
                  components,
                  total,
                  chunk_size,
                  num_workers):
    """
    Returns the calculated updates required to the entropies of the given
    ``contingencies`` and ``fragment_counts`` as a result of merging the
    fragments of each component in ``components``. This computation is
    performed in parallel with ``num_workers`` processes.
    """
    components_chunks = create_chunk_slices(len(components), chunk_size)
    delayed_delta_H_contingencies = [dask.delayed(delta_entropy_in_chunk)(contingencies,
                                                                          components[chunk],
                                                                          chunk,
                                                                          len(components),
                                                                          total)
                                     for chunk in components_chunks]
    delayed_delta_H_seg = [dask.delayed(delta_entropy_in_chunk)(fragment_counts,
                                                                components[chunk],
                                                                chunk,
                                                                len(components),
                                                                total)
                                     for chunk in components_chunks]
    delta_H_contingencies = dask.delayed(sum)(delayed_delta_H_contingencies).compute(num_workers=num_workers)
    delta_H_seg = dask.delayed(sum)(delayed_delta_H_seg).compute(num_workers=num_workers)
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
    logger.debug("Creating chunks of size {0} for {1} elements".format(chunk_size, total_size))
    return [slice(i, min(i+chunk_size, total_size)) for i in range(0, total_size, chunk_size)]

def segmentation_entropies(contingencies,
                           seg_counts,
                           gt_seg_counts,
                           total,
                           chunk_size,
                           num_workers):
    dask.config.set(scheduler='processes')
    contingencies_chunks = create_chunk_slices(len(contingencies), chunk_size)
    seg_counts_chunks = create_chunk_slices(len(seg_counts), chunk_size)
    gt_seg_counts_chunks = create_chunk_slices(len(gt_seg_counts), chunk_size)
    contingency_list = list(contingencies.values())
    seg_count_list = list(seg_counts.values())
    gt_seg_count_list = list(gt_seg_counts.values())

    delayed_H_contingencies = [
            dask.delayed(entropy_in_chunk)(
                contingency_list[c],
                total) for c in contingencies_chunks]
    delayed_H_seg = [
            dask.delayed(entropy_in_chunk)(
                seg_count_list[c],
                total) for c in seg_counts_chunks]
    delayed_H_gt_seg = [
            dask.delayed(entropy_in_chunk)(
                gt_seg_count_list[c],
                total) for c in gt_seg_counts_chunks]

    H_contingencies = dask.delayed(sum)(delayed_H_contingencies).compute(num_workers=num_workers)
    H_seg = dask.delayed(sum)(delayed_H_seg).compute(num_workers=num_workers)
    H_gt_seg = dask.delayed(sum)(delayed_H_gt_seg).compute(num_workers=num_workers)
    return (H_contingencies, H_seg, H_gt_seg)

def parallel_mergewise_score(rag,
                             fragments_file,
                             fragments_dataset,
                             gt_seg_file,
                             gt_seg_dataset,
                             total_roi,
                             block_size,
                             table_chunk_size,
                             component_chunk_size,
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
    
    logger.info("Calculating entropies for fragments")
    (H_contingencies, H_seg, H_gt_seg) = segmentation_entropies(contingencies,
                                                                fragment_counts,
                                                                gt_seg_counts,
                                                                total,
                                                                table_chunk_size,
                                                                num_workers)
    voi_split = H_contingencies - H_gt_seg
    voi_merge = H_contingencies - H_seg
    logger.info("Fragment VOI split: {0} VOI merge: {1}".format(voi_split, voi_merge))

    results = [] 
    for threshold in thresholds:
        logger.info("Calculating connected components for threshold {0}".format(threshold))
        components = rag.get_connected_components(threshold)
        (delta_H_contingencies, delta_H_seg) = delta_entropy(contingencies,
                                                             fragment_counts,
                                                             components,
                                                             total,
                                                             component_chunk_size,
                                                             num_workers)
        logger.info("Calculated updates are {0} and {1}".format(delta_H_contingencies,
                                                                 delta_H_seg))
        voi_split = (H_contingencies + delta_H_contingencies) - H_gt_seg
        voi_merge = (H_contingencies + delta_H_contingencies) - (H_seg + delta_H_seg)
        results.append((voi_split, voi_merge))
    return results
