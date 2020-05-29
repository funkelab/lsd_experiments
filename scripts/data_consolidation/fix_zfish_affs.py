import daisy
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def multiply_in_block(
        block,
        affs,
        multiplied_ds):

    logging.info('Cropping in block %s' %block.read_roi)
    affs = affs.to_ndarray(block.write_roi)

    #get max aff value
    affs_max = affs.max()

    #scale between 0 & 1
    normalized_affs = affs/affs_max

    #multiply x and y affs by z affs
    normalized_affs[1]*=normalized_affs[0]
    normalized_affs[2]*=normalized_affs[0]

    #rescale between 0 and 255 (or max value for block)
    rescaled_affs = np.array(normalized_affs*affs_max,dtype=affs.dtype)

    multiplied_ds[block.write_roi] = rescaled_affs

def multiply(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers):

    logging.info('Loading affs...')

    affs = daisy.open_ds(in_file, in_ds)

    total_roi = affs.roi

    read_roi = daisy.Roi((0,)*3, (18000, 16800, 16800))
    write_roi = read_roi

    logging.info('Creating cropped dataset...')

    multiplied_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    affs.voxel_size,
                    dtype=affs.dtype,
                    write_roi=write_roi,
                    num_channels=3)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: multiply_in_block(
            b,
            affs,
            multiplied_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    in_file = sys.argv[1]
    out_file = in_file
    in_ds = 'volumes/affs/s0'
    out_ds = 'volumes/affs_multiplied'
    num_workers = 32

    multiply(
            in_file,
            in_ds,
            out_file,
            out_ds,
            num_workers)



