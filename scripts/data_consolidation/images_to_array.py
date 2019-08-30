import daisy
import logging
import json
import numpy as np
import os
import sys
from skimage import io

def image_to_array(path_to_images):
    l = []

    for f in sorted(os.listdir(path_to_images)):

        print('Appending %s to list' %f)

        l.append(io.imread(os.path.join(path_to_images, f)))

    return np.stack(l, axis=0)

def write_in_block(
        block,
        array,
        ds):

    print('Writing data to block %s' %block.read_roi)

    array = array.to_ndarray(block.write_roi)

    ds[block.write_roi] = array

def write_to_ds(
        path_to_images,
        out_file,
        out_ds,
        num_workers):

    print('Converting images to numpy array...')

    array = image_to_array(path_to_images)

    voxel_size = [60,56,56]

    shape = [i*j for i,j in zip(list(array.shape), voxel_size)]

    total_roi = daisy.Roi((0,0,0), tuple(shape))

    array = daisy.Array(array, total_roi, daisy.Coordinate(voxel_size)) 

    read_roi = daisy.Roi((0, 0, 0), (2520, 2520, 2520))
    write_roi = daisy.Roi((0, 0, 0), (2520, 2520, 2520))

    print('Creating dataset...')

    ds = daisy.prepare_ds(
            out_file,
            out_ds,
            total_roi,
            voxel_size,
            dtype=array.dtype,
            write_roi=write_roi)

    print('Writing to dataset...')

    daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: write_in_block(
                b,
                array,
                ds),
            fit='shrink',
            num_workers=num_workers,
            read_write_conflict=False)


if __name__ == '__main__':

    write_to_ds(
            sys.argv[1],
            sys.argv[2],
            'volumes/raw',
            1)
