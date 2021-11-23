import daisy
import sys
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def crop_in_block(
        block,
        ds,
        cropped_ds):

    logging.info('Cropping in block %s' %block.read_roi)
    cropped = ds.to_ndarray(block.write_roi)

    cropped_ds[block.write_roi] = cropped

def crop(
        in_file,
        in_ds,
        out_file,
        out_ds,
        section,
        num_workers):

    logging.info('Loading ds...')

    try:
        ds = daisy.open_ds(in_file, in_ds)
    except:
        ds = daisy.open_ds(in_file, in_ds + '/s0')

    offset = ds.roi.get_begin()
    shape = ds.roi.get_shape()

    voxel_size = ds.voxel_size

    crop_roi = daisy.Roi(
                    (section*voxel_size[0],offset[1],offset[2]),
                    (voxel_size[0],shape[1],shape[2])
                )

    read_roi = crop_roi
    write_roi = crop_roi

    logging.info('Creating cropped dataset...')

    if 'raw' in in_ds:
        num_channels = 1

    else:
        num_channels = ds.shape[0]

    cropped_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    crop_roi,
                    voxel_size,
                    dtype=ds.dtype,
                    write_roi=write_roi,
                    num_channels=num_channels)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        crop_roi,
        read_roi,
        write_roi,
        process_function=lambda b: crop_in_block(
            b,
            ds,
            cropped_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    sections = [5862,6036,6918,6521,6657]

    vols = ['qual_1.zarr',
            'qual_3.zarr',
            'qual_4.zarr',
            'qual_5.zarr',
            'qual_5.zarr']

    setups = [
            'setup01',
            'setup01',
            'setup02',
            'setup03',
            'setup04',
            'setup05',
            'setup06',
            'setup07',
            'setup09']

    iterations = [
                    400,
                    400,
                    400,
                    400,
                    300,
                    300,
                    400,
                    200,
                    200]

    ds_names = [
            'raw',
            'vanilla',
            'mtlsd',
            'lsd',
            'auto_basic',
            'auto_full',
            'long_range',
            'malis',
            'mtlsd_malis']

    base_path = sys.argv[1]

    for a,b in zip(sections, vols):

        for c,d,e in zip(setups, iterations, ds_names):

            index = (sections.index(a)+1)

            in_file = os.path.join(base_path, c, str(d)+'000',b)

            out_file = os.path.join(
                    base_path,
                    'qualitative_samples',
                    'qual_%d.zarr'%index)

            if e == 'lsd':
                in_ds = 'volumes/lsds'
            elif e == 'raw':
                in_ds = 'volumes/raw'
            else:
                in_ds = 'volumes/affs'

            out_ds = e

            print(in_file)
            print(in_ds)
            print(out_file)
            print(out_ds)
            print('\n', '\n')

            crop(in_file, in_ds, out_file, out_ds, a, num_workers=1)




