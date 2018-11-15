import dask
import daisy
import lsd
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def apply_mask_in_block(block, vol, mask, out):
    logging.info("Applying mask in {0}".format(block))
    data = vol[block.read_roi].to_ndarray()
    mask_in_block = mask[block.read_roi].to_ndarray()
    data[mask_in_block == 0] = 0
    out[block.read_roi] = data

def parallel_apply_mask(
    """
    Apply a mask to a volume ONLY in the specified ROI.

    Args:

        in_file (``string``):

            The input h5-like file containing a volume to be masked.

        in_dataset (``string``):

            Name of dataset containing volume to be masked.

        mask_file (``string``):

            The h5-like file containing a mask.

        mask_dataset (``string``):

            Name of mask dataset. NOTE: the masking procedure will ONLY
            mask out regions in which mask is 0. Nonzero values in the mask
            are ignored.

        out_file (``string``):

            The output h5-like file to which a masked out volume will
            be written.

        out_dataset (``string``):

            Name of output masked dataset.

        offset (``tuple`` of ``int``):

            The offset in world units to start masking in.

        size (``tuple`` of ``int``):

            The size in world units of the desired ROI.
    """
        in_file,
        in_dataset,
        mask_file,
        mask_dataset,
        out_file,
        out_dataset,
        offset,
        total_size,
        block_size,
        num_workers,
        retry):
    
    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = daisy.Roi((0,)*3, block_size)
    vol = daisy.open_ds(in_file, in_dataset)
    if offset is None or total_size is None:
        total_roi = vol.roi
    else:
        total_roi = daisy.Roi(offset, total_size)
    mask = daisy.open_ds(mask_file, mask_dataset)
    
    logging.info("Constructing output dataset")
    out = daisy.prepare_ds(out_file,
                           out_dataset,
                           total_roi,
                           vol.voxel_size,
                           dtype=vol.dtype,
                           write_roi=write_roi)

    logging.info("Starting masking tasks")
    for i in range(retry + 1):
        if daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            lambda b: apply_mask_in_block(
                b,
                vol,
                mask,
                out),
            fit='shrink',
            num_workers=num_workers,
            processes=True,
            read_write_conflict=False):
                break

        if i < retry:
            logging.error("parallel masking failed, retrying %d/%d", i + 1, retry)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    parallel_apply_mask(**config)
