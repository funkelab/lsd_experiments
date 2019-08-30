import daisy
import json
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def mask_in_block(
        block,
        gt,
        masked_ds):

    logging.info('Masking in block %s' %block.read_roi)

    masked = gt.to_ndarray(block.write_roi) != 0

    masked_ds[block.write_roi] = masked

def mask(
        gt_file,
        gt_ds,
        out_file,
        out_ds,
        num_workers):

    logging.info('Loading mask and gt...')

    gt = daisy.open_ds(gt_file, gt_ds)

    total_roi = daisy.Roi((0,0,0), gt.roi.get_shape())

    read_roi = daisy.Roi((0, 0, 0), (4096, 4096, 4096))
    write_roi = daisy.Roi((0, 0, 0), (4096, 4096, 4096))

    logging.info('Creating cropped dataset...')

    masked_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    gt.voxel_size,
                    np.uint8,
                    write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: mask_in_block(
            b,
            gt,
            masked_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    mask(
            sys.argv[1],
            'volumes/labels/mask_ids/s0',
            sys.argv[2],
            'volumes/labels/test_new_mask',
            num_workers=40)
