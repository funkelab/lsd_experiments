import daisy
import sys
import json
import logging
import numpy as np
from lsd.local_shape_descriptor import get_local_shape_descriptors as glsd

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def calc_in_block(
        block,
        seg,
        lsd_ds):

    logging.info('Calculating in block %s' %block.read_roi)
    seg_array = seg.to_ndarray(block.write_roi)

    computed_lsds = glsd(
                        seg_array,
                        sigma=(80,)*3,
                        voxel_size=seg.voxel_size,
                        downsample=2)
    print(computed_lsds)

    lsd_ds[block.write_roi] = computed_lsds

def calculate_lsds(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers):

    logging.info('Loading seg...')

    seg = daisy.open_ds(in_file, in_ds)
    total_roi = seg.roi
    voxel_size = seg.voxel_size

    read_roi = daisy.Roi((0,)*3, (736,)*3)
    write_roi = read_roi

    logging.info('Creating cropped dataset...')

    lsd_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    voxel_size,
                    dtype=np.float32,
                    write_roi=write_roi,
                    num_channels=10)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: calc_in_block(
            b,
            seg,
            lsd_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)

if __name__ == '__main__':

    in_file = sys.argv[1]
    in_ds = 'volumes/segmentation/s0'
    out_file = in_file
    out_ds = 'volumes/calculated_lsds'
    num_workers = 10

    calculate_lsds(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers)
