import daisy
import json
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def mask_in_block(
        block,
        affs,
        filter_value,
        masked_ds):

    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logging.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float32)
    else:
        max_affinity_value = 1.0

    average_affs = np.mean(affs.data[0:2]/max_affinity_value, axis=0)

    print("average affs: ", average_affs)

    masked = affs.to_ndarray(block.write_roi)[0,:,:,:]

    masked[average_affs<filter_value]=0
    masked[average_affs>filter_value]=1

    masked_ds[block.write_roi] = masked

def create_mask(
        in_file,
        in_ds,
        out_file,
        out_ds,
        filter_value,
        num_workers):

    logging.info('Loading ds...')

    ds = daisy.open_ds(in_file, in_ds)

    roi = ds.roi

    read_roi = daisy.Roi((0,)*3, (13950, 9300, 9300))
    write_roi = read_roi

    logging.info('Creating mask dataset...')

    masked_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    roi,
                    ds.voxel_size,
                    dtype='uint8',
                    write_size=write_roi.get_shape(),
                    num_channels=1)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        roi,
        read_roi,
        write_roi,
        process_function=lambda b: mask_in_block(
            b,
            ds,
            filter_value,
            masked_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    in_file = sys.argv[1]
    in_ds = 'volumes/affs/s0'
    out_file = in_file
    out_ds = 'volumes/boundary_mask_2'
    filter_value = 0.1
    num_workers = 20

    create_mask(
        in_file,
        in_ds,
        out_file,
        out_ds,
        filter_value,
        num_workers)
