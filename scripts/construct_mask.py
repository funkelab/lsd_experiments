import json
import logging
import numpy as np
import os
import daisy
import sys
import csv
from skimage import measure
from scipy.ndimage import morphology

logging.basicConfig(level=logging.DEBUG)

def first_pass(block, vol, mask):
    logging.debug("First pass in {0}".format(block.read_roi))
    shape = np.array((block.read_roi / vol.voxel_size).get_shape())
    data = vol[block.read_roi].to_ndarray()

    if not np.any(data):
        mask[block.write_roi] = np.ones(shape, dtype=np.uint8)

def second_pass(block, vol, mask, erosion_size):
    logging.debug("Second pass in {0}, writing to {1}".format(block.read_roi, block.write_roi))
    distance = np.linalg.norm(erosion_size)
    read_shape = np.array((block.read_roi / vol.voxel_size).get_shape())
    write_shape = np.array((block.write_roi / vol.voxel_size).get_shape())
    dims = block.read_roi.dims()
    data = vol[block.read_roi].to_ndarray()
    known_background = data == 1
    mask_in_block = np.full(read_shape, 2, dtype=np.uint8)
    
    if np.any(known_background):
        logging.debug("{0} is a boundary region".format(block.write_roi))
        mask_in_block[known_background] = 1
        zero_map = measure.label(np.uint8(data == 0))
        overlaps = np.logical_and(zero_map, known_background)
        index = np.unravel_index(np.argmax(overlaps), overlaps.shape)
        background_id = zero_map[index]
        true_background = zero_map == background_id
        true_foreground = np.logical_not(true_background)
        true_background = morphology.distance_transform_edt(true_foreground)
        true_background = true_background < distance
        mask_in_block[true_background] = 1
    else:
        logging.debug("{0} lies completely in interior".format(block.write_roi))
    
    block_in_world = daisy.Array(mask_in_block, block.read_roi, vol.voxel_size)
    mask[block.write_roi] = block_in_world[block.write_roi]

def binarize(block, mask):
    logging.debug("Binarizing mask in {0}".format(block.read_roi))

def construct_mask(
        in_file,
        to_mask,
        out_file,
        erosion_size,
        mask_context,
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
    read_roi = daisy.Roi((0,)*vol.roi.dims(),
                         [block_size[i] + mask_context[i]
                             for i in range(vol.roi.dims())])
    write_roi = daisy.Roi((0,)*vol.roi.dims(), block_size)

    logging.debug("Constructing mask dataset in {0}".format(out_file))
    mask = daisy.prepare_ds(out_file, 'volumes/labels/mask', vol.roi, vol.voxel_size, dtype=np.uint8)
    logging.info("Masking volume in {0}".format(in_file))

    # mask blocks in parallel
    logging.info("Starting first pass")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            write_roi,
            write_roi,
            lambda b: first_pass(
                b,
                vol,
                mask),
            fit='shrink',
            processes=True,
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("first pass failed, retrying %d/%d", i + 1, retry)
    
    logging.info("Starting second pass")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: second_pass(
                b,
                vol,
                mask,
                erosion_size),
            fit='shrink',
            processes=True,
            num_workers=num_workers,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("second pass failed, retrying %d/%d", i + 1, retry)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    construct_mask(**config)
