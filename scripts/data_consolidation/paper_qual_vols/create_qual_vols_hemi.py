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

    g = daisy.Coordinate((8,1440,1440))

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

    read_roi = daisy.Roi((0, 0, 0), (16,2880,2880))
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

    centers = [
                (146696, 215032, 203024),
                (148936, 208168, 206264),
                (148376, 211144, 207400),
                (161144, 208784, 184616),
                (164440, 206936, 192320)
            ]

    setups = [
            'setup01',
            'setup02',
            'setup03',
            'setup04',
            'setup05',
            'setup12',
            'setup25',
            '01_data']

    iterations = [400, 400, 200, 200, 200, 400, 400, None]

    ds_names = [
            'vanilla',
            'mtlsd',
            'lsd',
            'auto_basic',
            'auto_full',
            'long_range',
            'malis',
            'raw']

    net_base_path = sys.argv[1]
    raw_base_path = sys.argv[2]

    for center in centers:

        for a,b,c in zip(setups, iterations, ds_names):

            index = (centers.index(center)+1)
            if index<=3:
                vol=1
            else:
                vol=3

            if c is 'raw':
                data_path = os.path.join(raw_base_path, a)
                in_file = os.path.join(data_path, 'hemi_testing_roi_%d.zarr'%vol)
            else:
                data_path = os.path.join(net_base_path, a)
                net_path = os.path.join(data_path, str(b)+'000')
                in_file = os.path.join(net_path, 'cropout_%d.zarr'%vol)

            out_file = os.path.join(
                    net_base_path,
                    'qualitative_samples',
                    'qual_%d.zarr'%index)

            if c == 'lsd':
                in_ds = 'volumes/lsds'
            elif c == 'raw':
                in_ds = 'volumes/raw'
            else:
                in_ds = 'volumes/affs'

            out_ds = c

         #    print(in_file)
            # print(in_ds)
            # print(out_file)
            # print(out_ds)
            # print('\n', '\n')

            crop(in_file, in_ds, out_file, out_ds, center, num_workers=1)




