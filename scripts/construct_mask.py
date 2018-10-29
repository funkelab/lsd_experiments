import json
import logging
import numpy as np
import os
import daisy
import sys
import csv
from skimage import measure
from scipy.ndimage import filters, morphology

logging.basicConfig(level=logging.DEBUG)

def touching_edge(inds, shape):
    if np.any(inds == 0):
        return True
    for d in range(len(shape)):
        if np.any(inds[d] == shape[d]-1):
            return True
    return False

def mask_in_block(block, vol, mask):
    logging.debug("Masking in {0}".format(block))
    read_shape = np.array((block.read_roi / vol.voxel_size).get_shape())
    dims = block.read_roi.dims()
    data = filters.gaussian_filter(vol.to_ndarray(block.read_roi, 0), 3)
    
    mask_in_block = np.ones(read_shape, dtype=np.uint8)
    potential_background = measure.label(np.uint8(data == 0))
    background_props = measure.regionprops(potential_background)

    for region in background_props:
        coords = region.coords
        inds = np.moveaxis(coords, -1, 0)
        if touching_edge(inds, read_shape):
            mask_in_block[tuple(inds)] = 0

    block_in_world = daisy.Array(mask_in_block, block.read_roi, vol.voxel_size)
    mask[block.write_roi] = block_in_world[block.write_roi]

def erode_in_block(block, mask, erosion_size):
    logging.debug("Eroding in {0}".format(block))
    # distance is padded to counteract dilation from blurring
    distance = np.linalg.norm(erosion_size) + np.sqrt(27)
    mask_in_block = mask[block.read_roi].to_ndarray()
    if (not np.all(mask_in_block)) and np.any(mask_in_block == 0):
        logging.debug("Boundary in {0}".format(block))
        distances = morphology.distance_transform_edt(mask_in_block)
        mask_in_block = np.uint8(distances >= distance)
        mask[block.write_roi] = mask_in_block

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
    total_roi = vol.roi.grow(daisy.Coordinate(mask_context),
                             daisy.Coordinate(mask_context))
    write_roi = daisy.Roi((0,)*vol.roi.dims(), block_size)
    read_roi = write_roi.grow(daisy.Coordinate(mask_context), daisy.Coordinate(mask_context))

    logging.debug("Constructing mask dataset in {0}".format(out_file))
    mask = daisy.prepare_ds(out_file, 'volumes/labels/mask', vol.roi, vol.voxel_size, dtype=np.uint8, write_roi = write_roi)
    logging.info("Masking volume in {0}".format(in_file))
    
    # mask blocks in parallel
    logging.info("Starting first pass")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: mask_in_block(
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
    
    # erode blocks in parallel
    logging.info("Starting second pass")
    for i in range(retry + 1):
        # TODO: check function
        if daisy.run_blockwise(
            mask.roi,
            read_roi,
            read_roi,
            lambda b: erode_in_block(
                b,
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
