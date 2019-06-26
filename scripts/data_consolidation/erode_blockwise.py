import daisy
import json
import logging
import numpy as np
import sys

from daisy import Array
from scipy import ndimage

logging.basicConfig(level=logging.INFO)

def erode_in_block(
        block,
        gt,
        gt_ds,
        iterations,
        border_value):

    gt = gt.to_ndarray(block.write_roi)

    logging.info('Eroding in block: %s', block.write_roi)

    foreground = np.zeros(shape=gt.shape, dtype=bool)

    for label in np.unique(gt):
        if label == 0:
            continue
        label_mask = gt==label

        eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=iterations, border_value=border_value)
        foreground = np.logical_or(eroded_label_mask, foreground)

    background = np.logical_not(foreground)
    gt[background] = 0

    gt_ds[block.write_roi] = gt

def erode(
        in_file,
        out_file,
        in_ds,
        out_ds,
        iterations,
        border_value,
        num_workers):

    logging.info('Loading gt data...')

    gt = daisy.open_ds(in_file, in_ds)

    read_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))
    write_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))

    logging.info('Creating eroded dataset...')

    gt_ds = daisy.prepare_ds(
            out_file,
            out_ds,
            gt.roi,
            gt.voxel_size,
            gt.data.dtype,
            write_roi)

    daisy.run_blockwise(
        gt.roi,
        read_roi,
        write_roi,
        process_function=lambda b: erode_in_block(
            b,
            gt,
            gt_ds,
            iterations,
            border_value),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)

if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    erode(**config)
