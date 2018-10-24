import json
import logging
import numpy as np
import os
import daisy
import sys
import csv
from scipy.ndimage import morphology

logging.basicConfig(level=logging.DEBUG)

def construct_mask_in_block(block, vol, mask, context):
    logging.debug("Masking in {0}".format(block.read_roi))
    shape = np.array((block.read_roi / vol.voxel_size).get_shape())
    data = vol[block.read_roi].to_ndarray()
    mask_in_block = np.ones(shape, dtype=np.uint16)
    mask_in_block[data == 0] = 0
    if not np.all(mask_in_block == 1) and np.count_nonzero(mask_in_block) > 0:
        mask_in_block = morphology.binary_dilation(mask_in_block, structure=np.ones((15, 15, 15))).astype(np.uint16)
        mask_in_block = morphology.binary_erosion(
                mask_in_block,
                structure=np.ones(np.uint32((context*2 + 1)/15)),
                iterations=15,
                border_value=1).astype(np.uint16)
        mask_in_block = morphology.binary_erosion(
                mask_in_block,
                structure=np.ones((15, 15, 15)),
                iterations=15,
                border_value=1).astype(np.uint16)
        mask[block.write_roi] = mask_in_block
    elif np.all(mask_in_block == 1):
        mask[block.write_roi] = np.ones(shape, dtype=np.uint16)
        logging.debug("Block {0} is fully interior".format(block.read_roi))
    logging.debug("Finished block {0}".format(block.read_roi))

def construct_mask(
        in_file,
        to_mask,
        out_file,
        context,
        block_size,
        num_workers,
        retry):
    '''Constructs mask for desired ROI.

    Args:

        in_file (``string``):

            Path of the file containing region being masked.

        to_mask (``string``):

            Name of dataset containing region being masked.

        out_file (``string``):

            Path of the file to which mask dataset will be added.

        context (``tuple`` of ``int``):

            The shape of the network context (in voxels!), which will be
            used to further erode the mask.

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        num_channels (``int``):

            The number of channels in the volume (default 1).

        num_workers (``int``):

            How many blocks to run in parallel.

        retry (``int``):

            How many times to retry failed tasks.
    '''

    logging.info("Reading volume from {0} with dataset {1}".format(in_file, to_mask))

    vol = daisy.open_ds(in_file, to_mask)
    total_roi = vol.roi
    read_roi = daisy.Roi((0,)*vol.roi.dims(), block_size)
    write_roi = daisy.Roi((0,)*vol.roi.dims(), block_size)

    logging.debug("Constructing mask dataset in {0}".format(out_file))
    mask = daisy.prepare_ds(out_file, 'volumes/labels/mask', vol.roi, vol.voxel_size, dtype=np.uint16)
    logging.info("Masking volume in {0}".format(in_file))

    # mask blocks in parallel
    logging.info("Starting masking tasks")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: construct_mask_in_block(
                b,
                vol,
                mask,
                np.array(context)),
            fit='shrink',
            processes=True,
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel mask failed, retrying %d/%d", i + 1, retry)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    construct_mask(**config)
