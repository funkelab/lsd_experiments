import json
import logging
import numpy as np
import os
import daisy
import sys
import csv

logging.basicConfig(level=logging.DEBUG)

def scan(
        in_file,
        in_ds,
        out_file,
        block_size,
        num_workers,
        retry):
    '''Scans large multidimensional datasets blockwise in parallel to gather
    desired data about each block.

    Args:

        in_file (``string``):

            Name of the file from which data will be read and searched through.

        in_ds (``string``):

            Name of the dataset in the file to be searched.

        out_file (``string``):

            Name of the file in which the results of the search will be output.

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        num_workers (``int``):

            How many blocks to run in parallel.

        retry (``int``):

            How many times to retry failed tasks.
    '''

    logging.info("Reading volume from {0} with dataset {1}".format(in_file, in_ds))

    vol = daisy.open_ds(in_file, in_ds, mode='r')

    # prepare results storage
    # TODO: use a database maybe?
    results = set()
    logging.info("Storing all results to {0}".format(out_file))
    total_roi = vol.roi
    read_roi = daisy.Roi((0,)*vol.roi.dims(), block_size)
    write_roi = daisy.Roi((0,)*vol.roi.dims(), block_size)

    # scan blocks in parallel
    logging.info("Starting scan tasks")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: nonzero_in_block(
                vol,
                b,
                results),
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel search failed, retrying %d/%d", i + 1, retry)

    # write results to file
    print("Storing {0} results".format())
    with open(out_file, 'w') as f:
        [f.write(r + "\n") for r in results]

def nonzero_in_block(vol, block, results):
    """
    Stores block ROI string and proportion nonzero elements in block to
    results.
    """
    vol_data = np.uint64(vol.intersect(block.read_roi).data)
    logging.debug("Scanning in block {0}".format(block.read_roi))
    # determine proportion of zero values in block
    num_nonzero = np.float64(np.count_nonzero(vol_data))
    num_total = np.float64(vol_data.size)
    proportion_nonzero = (num_nonzero / num_total)
    # store block results
    results.add("{0}, {1}".format(block.read_roi, proportion_nonzero))
    logging.info("Block {0}: {1}".format(block.read_roi, proportion_nonzero))

def block_done(seg_out, block, whitelist):
    # TODO
    return False

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    scan(**config)
