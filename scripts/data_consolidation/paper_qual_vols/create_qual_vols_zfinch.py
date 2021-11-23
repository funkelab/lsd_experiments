import daisy
import sys
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def generate_roi(center):

    center = daisy.Coordinate(center)

    roi = daisy.Roi(center, (0,)*3)

    g = daisy.Coordinate((20,1440,1440))

    roi = roi.grow(g, g)

    return roi

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
        center,
        num_workers):

    logging.info('Loading ds...')

    try:
        ds = daisy.open_ds(in_file, in_ds)
    except:
        ds = daisy.open_ds(in_file, in_ds + '/s0')

    crop_roi = generate_roi(center)

    read_roi = daisy.Roi((0, 0, 0), (40,2880,2880))
    write_roi = read_roi

    logging.info('Creating cropped dataset...')

    if 'raw' in in_ds:
        num_channels = 1

    else:
        num_channels = ds.shape[0]

    cropped_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    crop_roi,
                    ds.voxel_size,
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

    in_file = sys.argv[1]
    in_ds = 'volumes/affs'
    out_file = 'test.zarr'
    out_ds = 'lr_affs'
    center = (57020,53019,55350)
    num_workers = 1

    centers = [
                (57020,53019,55350),
                (59940,51354,29394),
                (53020,51363,76383),
                (73220,14067,45864),
                (79040,19710,69183)
            ]

    setups = [
            'setup01',
            'setup02',
            'setup03',
            'setup04',
            'setup05',
            'setup06',
            'setup07',
            'setup08',
            'zebrafinch_realigned.zarr']

    iterations = [400, 400, 400, 163, 190, 400, 400, 400, None]

    ds_names = [
            'vanilla',
            'mtlsd',
            'lsd',
            'auto_basic',
            'auto_full',
            'long_range',
            'malis',
            'mtlsd_malis',
            'raw']

    base_path = sys.argv[1]

    for center in centers:

        for a,b,c in zip(setups, iterations, ds_names):

            data_path = os.path.join(base_path, a)

            net_path = os.path.join(data_path, str(b)+'000')

            if 'z' in a:
                in_file = data_path
            else:
                in_file = os.path.join(net_path, 'zebrafinch.zarr')

            out_file = os.path.join(
                    base_path,
                    'qualitative_samples',
                    'qual_%d.zarr'%(centers.index(center)+1))

            if c == 'lsd':
                in_ds = 'volumes/lsds'
            elif c == 'raw':
                in_ds = 'volumes/raw'
            else:
                in_ds = 'volumes/affs'

            out_ds = c

            crop(in_file, in_ds, out_file, out_ds, center, num_workers)




