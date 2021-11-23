import daisy
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def create_mask_in_block(
        block,
        ds,
        masked_ds):

    logging.info('Cropping in block %s' %block.read_roi)
    masked = ds.to_ndarray(block.write_roi)
    masked[masked<254]=1
    masked[masked>=254]=0

    masked_ds[block.write_roi] = masked

def create_mask(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers):

    logging.info('Loading ds...')

    ds = daisy.open_ds(in_file, in_ds)

    # roi = daisy.Roi((0,)*3, (1000,5000,5000))
    roi = ds.roi

    read_roi = daisy.Roi((0,)*3, (15000,600,600))
    write_roi = read_roi

    logging.info('Creating mask dataset...')

    masked_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    roi,
                    ds.voxel_size,
                    dtype=ds.dtype,
                    write_size=write_roi.get_shape(),
                    num_channels=1)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        roi,
        read_roi,
        write_roi,
        process_function=lambda b: create_mask_in_block(
            b,
            ds,
            masked_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    in_file = sys.argv[1]
    in_ds = 'volumes/raw/s0'
    out_file = in_file
    out_ds = 'volumes/boundary_mask'
    num_workers = 20

    create_mask(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers)
