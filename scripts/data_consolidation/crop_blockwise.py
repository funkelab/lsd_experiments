import daisy
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def crop_in_block(
        block,
        seg,
        cropped_ds):

    logging.info('Cropping in block %s' %block.read_roi)
    cropped = seg.to_ndarray(block.write_roi)

    cropped_ds[block.write_roi] = cropped

def crop(
        in_file,
        in_ds,
        out_file,
        out_ds,
        roi_offset,
        roi_shape,
        num_workers):

    logging.info('Loading seg...')

    seg = daisy.open_ds(in_file, in_ds)

    crop_roi = daisy.Roi((roi_offset), (roi_shape))

    read_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))
    write_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))

    logging.info('Creating cropped dataset...')

    cropped_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    crop_roi,
                    seg.voxel_size,
                    dtype=seg.dtype,
                    write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        crop_roi,
        read_roi,
        write_roi,
        process_function=lambda b: crop_in_block(
            b,
            seg,
            cropped_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    crop(**config)


