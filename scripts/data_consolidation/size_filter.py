import daisy
import json
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def filter_in_block(
        block,
        seg,
        filtered_ds,
        size_filter):

    logging.info('Cropping in block %s' %block.read_roi)
    labels = seg.to_ndarray(block.write_roi)

    f = []

    u = np.unique(labels, return_counts=True)

    for i,j in zip(u[0],u[1]):
        if i == 0:
            continue
        if j < size_filter:
            f.append(i)
        else:
            pass

    labels[np.isin(labels, f, invert=False)] = 0

    filtered_ds[block.write_roi] = labels

def sf(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers,
        block_size,
        size_filter):

    logging.info('Loading seg...')

    seg = daisy.open_ds(in_file, in_ds)

    total_roi = seg.roi

    read_roi = daisy.Roi((0,)*3, block_size)
    write_roi = read_roi

    logging.info('Creating filtered dataset...')

    filtered_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    seg.voxel_size,
                    dtype=seg.dtype,
                    write_size=write_roi.get_shape(),
                    num_channels=1)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: filter_in_block(
            b,
            seg,
            filtered_ds,
            size_filter),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    in_file = sys.argv[1]
    in_ds = sys.argv[2]
    out_file = in_file
    out_ds = 'volumes/segmentation_size_filter_1000'
    num_workers = 20
    block_size = [21000, 1400, 1400]
    size_filter = 1000

    sf(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers,
        block_size,
        size_filter)
