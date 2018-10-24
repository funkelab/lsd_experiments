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
        seg_counts_shape,
        gt_seg_counts_shape,
        contingencies_shape,
        ignore=[0]):
    logging.debug("Calculating contingencies in {0}".format(block.read_roi))
    block_id = block.block_id
    seg_in_block =  seg[block.read_roi].to_ndarray()
    gt_seg_in_block = gt_seg[block.read_roi].to_ndarray()
    seg_indices = np.ravel(seg_in_block)
    gt_seg_indices = np.ravel(gt_seg_in_block)
    ignored = np.logical_or(np.isin(seg_indices, ignore), np.isin(gt_seg_indices, ignore))
    data = np.ones(seg_indices.shape)
    data[ignored] = 0
    partial_contingencies = sparse.coo_matrix(
            (data, (seg_indices, gt_seg_indices)),
            shape=contingencies_shape).tocsc()
    partial_seg_counts = sparse.coo_matrix(
            (data, (seg_indices, np.zeros(seg_indices.shape))),
            shape=seg_counts_shape).tocsc()
    partial_gt_seg_counts = sparse.coo_matrix(
            (data, (np.zeros(seg_indices.shape), gt_seg_indices)),
            shape=gt_seg_counts_shape).tocsc()
    contingencies.append(partial_contingencies)
    seg_counts.append(partial_seg_counts)
    gt_seg_counts.append(partial_gt_seg_counts)
    totals.append(np.sum(data))

def entropy_in_chunk(sparse_chunk, total):
    # assumes sparse_chunk is nonzero
    probabilities = sparse_chunk / total
    return np.sum(-probabilities * np.log2(probabilities))

def create_chunk_slices(total_size, chunk_size):
    logging.debug("Creating chunks of size {0} for {1} elements".format(chunk_size, total_size))
    return [slice(i, min(i+chunk_size, total_size)) for i in range(0, total_size, chunk_size)]

def parallel_score(
        tmp_fname,
        tmp_seg_name,
        tmp_gt_seg_name,
        total_roi,
        block_size,
        chunk_size,
        seg_counts_shape,
        gt_seg_counts_shape,
        contingencies_shape,
        num_workers,
        retry):
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)

    seg = daisy.open_ds(tmp_fname, tmp_seg_name)
    gt_seg = daisy.open_ds(tmp_fname, tmp_gt_seg_name)
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

def main():
    if len(argv) != 4+1:
        print("usage: {0} <gt_fname> <gt_dsname> <seg_fname> <seg_ds>".format(argv[0]))
        exit()

    _, gt_fname, gt_dsname, seg_fname, seg_ds = argv

    seg = daisy.open_ds(seg_fname, seg_ds)
    gt_seg = daisy.open_ds(gt_fname, gt_dsname)
    total_roi = gt_seg.roi
    block_size = (1024, 1024, 1024)
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    chunk_size = 16384
    num_workers = 12
    retry = 2
    seg_counts_shape = (int(10e7), 1)
    gt_seg_counts_shape = (1, int(10e7))
    contingencies_shape = (int(10e7), int(10e7))
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

if __name__ == '__main__':
    main()
