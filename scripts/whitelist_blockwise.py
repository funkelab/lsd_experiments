import json
import logging
import numpy as np
import os
import daisy
import sys
import csv

logging.basicConfig(level=logging.DEBUG)

def whitelist(
        whitelist_file,
        in_file,
        out_file,
        block_size,
        num_workers,
        retry):
    '''Whitelist neurons in parallel blocks. Requires that a segmentation exists.

    Args:

        whitelist_file (``string``):

            Path of .csv file containing whitelisted neurons.

        in_file (``string``):

            Name of the file from which raw segmentations will be read.

        out_file (``string``):

            Name of the file in which whitelisted segmentations will be output.

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        num_workers (``int``):

            How many blocks to run in parallel.

        retry (``int``):

            How many times to retry failed tasks.

    '''

    assert whitelist_file is not None

    logging.info("Reading whitelisted neurons from {0}".format(whitelist_file))

    with open(whitelist_file, 'r') as f:
        reader = csv.reader(f)
        whitelist = [np.uint64(row[0]) for row in reader if 'traced' in row[-1].lower()]

    logging.info("Reading segmentations from {0}".format(in_file))
    seg_ds = "volumes/labels/neuron_ids"
    whitelisted_seg_ds = "volumes/labels/whitelisted"
    seg = daisy.open_ds(in_file, seg_ds, mode='r')

    # prepare whitelisted segmentation dataset
    seg_out = daisy.prepare_ds(
             out_file,
             whitelisted_seg_ds,
             seg.roi,
             seg.voxel_size,
             np.uint64,
             daisy.Roi((0, 0, 0), block_size))
    logging.info("Storing whitelisted segmentations to {0}".format(out_file))
    total_roi = seg.roi
    read_roi = daisy.Roi((0,)*seg.roi.dims(), block_size)
    write_roi = daisy.Roi((0,)*seg.roi.dims(), block_size)

    # remove nonwhitelisted neurons in parallel
    logging.info("Starting whitelisting tasks")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: whitelist_in_block(
                seg,
                b,
                seg_out,
                whitelist),
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel whitelisting failed, retrying %d/%d", i + 1, retry)

def whitelist_in_block(seg, block, seg_out, whitelist):
    seg_data = np.uint64(seg.intersect(block.read_roi).data)
    logging.debug("Sum of values in block {0} before whitelisting: {1}".format(block.read_roi, np.sum(seg_data)))
    logging.info("Whitelisting in block {0}".format(block.read_roi))
    # remove nonwhitelisted neurons from current block
    seg_data[np.isin(seg_data, whitelist, invert=True)] = 0
    logging.debug("Sum of values in block {0} after whitelisting: {1}".format(block.read_roi, np.sum(seg_data)))
    seg_out[block.write_roi] = daisy.Array(seg_data, block.write_roi, seg.voxel_size)

def block_done(seg_out, block, whitelist):
    # TODO
    return False

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    whitelist(**config)
