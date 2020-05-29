import daisy
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def fix_in_block(
        block,
        mask_in,
        mask_out):

    logging.info('Fixing in block %s' %block.read_roi)
    fixed = mask_in.to_ndarray(block.write_roi)

    mask_out[block.write_roi] = fixed >= 0.5

def fix_mask(
        in_file,
        in_ds,
        out_file,
        out_ds,
        num_workers,
        roi_offset=None,
        roi_shape=None):

    logging.info('Loading seg...')

    mask_in = daisy.open_ds(in_file, in_ds)

    if roi_offset:
        total_roi = daisy.Roi((roi_offset), (roi_shape))

    else:
        total_roi = mask_in.roi

    logging.info('Total roi %s' %total_roi)

    read_roi = daisy.Roi((0, 0, 0), (18000, 16800, 16800))
    write_roi = read_roi

    logging.info('Creating fixed dataset...')

    mask_out = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    mask_in.voxel_size,
                    dtype=mask_in.dtype,
                    write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: fix_in_block(
            b,
            mask_in,
            mask_out),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    fix_mask(**config)


