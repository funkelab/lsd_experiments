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
from parallel_contingencies_map import parallel_contingencies_map
from parallel_contingencies_rag import parallel_contingencies_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def fragment_entropies(contingencies,
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

def segmentation_entropies(contingencies,
                           seg_counts,
                           total,
                           chunk_size,
                           num_workers):
    dask.config.set(scheduler='processes')
    contingencies_chunks = create_chunk_slices(len(contingencies), chunk_size)
    seg_counts_chunks = create_chunk_slices(len(seg_counts), chunk_size)
    gt_seg_counts_chunks = create_chunk_slices(len(gt_seg_counts), chunk_size)
    contingency_list = list(contingencies.values())
    seg_count_list = list(seg_counts.values())

    delayed_H_contingencies = [
            dask.delayed(entropy_in_chunk)(
                contingency_list[c],
                total) for c in contingencies_chunks]
    delayed_H_seg = [
            dask.delayed(entropy_in_chunk)(
                seg_count_list[c],
                total) for c in seg_counts_chunks]

    H_contingencies = dask.delayed(sum)(delayed_H_contingencies).compute(num_workers=num_workers)
    H_seg = dask.delayed(sum)(delayed_H_seg).compute(num_workers=num_workers)
    return (H_contingencies, H_seg)

def parallel_componentwise_score(rag,
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
     total) = parallel_contingencies_map(fragments_file,
                                         fragments_dataset,
                                         gt_seg_file,
                                         gt_seg_dataset,
                                         total_roi,
                                         block_size,
                                         num_workers,
                                         retry)
    
    logger.info("Calculating entropies for fragments")
    (H_contingencies, H_seg, H_gt_seg) = fragment_entropies(contingencies,
                                                            fragment_counts,
                                                            gt_seg_counts,
                                                            total,
                                                            chunk_size,
                                                            num_workers)
    voi_split = H_contingencies - H_gt_seg
    voi_merge = H_contingencies - H_seg
    logger.info("Fragment VOI split: {0} VOI merge: {1}".format(voi_split, voi_merge))

    results = [] 
    for threshold in thresholds:
        logger.info("Calculating connected components for threshold {0}".format(threshold))
        components = rag.get_connected_components(threshold)
    
        logger.info("Calculating entropies for threshold {0}".format(threshold))

        (contingencies,
         seg_counts,
         total) = parallel_contingencies_rag(components,
                                             fragments_file,
                                             fragments_dataset,
                                             gt_seg_file,
                                             gt_seg_dataset,
                                             total_roi,
                                             block_size,
                                             num_workers,
                                             retry)

        (H_contingencies, H_seg) = segmentation_entropies(contingencies,
                                                          seg_counts,
                                                          total,
                                                          chunk_size,
                                                          num_workers)

        voi_split = H_contingencies - H_gt_seg
        voi_merge = H_contingencies - H_seg
        logger.info("Threshold {0} VOI split: {1} VOI merge: {2}".format(threshold, voi_split, voi_merge))
        results.append((voi_split, voi_merge))
    return results
