import daisy
import json
import logging
import numpy as np
import sys
from skimage.morphology import remove_small_objects

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def seg_in_block(
        block,
        pred,
        seg_ds):

    logging.info('Creating seg in block %s' %block.read_roi)
    pred = pred.to_ndarray(block.write_roi)

    thresh = pred >= 0.5

    seg = remove_small_objects(thresh.astype(bool)).astype(np.uint64)

    seg_ds[block.write_roi] = seg

def seg(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers):

    logging.info('Loading pred...')

    pred = daisy.open_ds(in_file, in_ds)

    total_roi = pred.roi

    read_roi = daisy.Roi((0,)*3, (13950, 9300, 9300))
    write_roi = daisy.Roi((0,)*3, (13950, 9300, 9300))

    logging.info('Creating seg dataset...')

    seg_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    pred.voxel_size,
                    dtype=np.uint64,
                    write_size=write_roi.get_shape(),
                    num_channels=1)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: seg_in_block(
            b,
            pred,
            seg_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    in_file = sys.argv[1]
    in_ds = sys.argv[2]
    out_file = in_file
    out_ds = 'volumes/sdt_seg'
    num_workers = 20

    seg(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers)
