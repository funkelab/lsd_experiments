import numpy as np
import os
import sys
import daisy
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def mask_neurons(
        mask_file,
        mask_ds,
        in_file,
        out_file,
        in_ds,
        out_ds,
        num_workers):

    '''Filter neurons in volume given a mask.

    Args:

        mask_file (``string``):

            Path of file containing mask

        mask_ds (``string``):

            Name of mask dataset

        in_file (``string``):

            Path of data file in which the segmentation to filter is contained

        out_file  (``string``):

            Path of file in which the filtered segmentation should be written to

        in_ds (``string``):

            Name of input unfiltered segmentation dataset

        out_ds (``string``):

            Name of output filtered segmentation dataset

        num_workers (``int``):

            Number of blocks to run in parallel

    '''

    #load mask to use for filtering
    mask = daisy.open_ds(mask_file, mask_ds)
    # print('Loaded mask')

    #load neuron dataset to filter
    seg = daisy.open_ds(in_file, in_ds)
    print('Loaded neuron ids')

    read_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))
    write_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))

    print('Read roi is %s, write roi is %s' %(read_roi, write_roi))

    #prepare the masked id dataset
    print('Preparing out dataset...')
    masked_ds = daisy.prepare_ds(
            out_file,
            out_ds,
            seg.roi,
            seg.voxel_size,
            np.uint64,
            write_roi)

    daisy.run_blockwise(
            seg.roi,
            read_roi,
            write_roi,
            process_function=lambda b: mask_in_block(
                b,
                seg,
                masked_ds,
                mask),
            fit='shrink',
            num_workers=num_workers,
            read_write_conflict=False)

def upsample(a, factor):

    for d, f in enumerate(factor):
        a = np.repeat(a, f, axis=d)

    return a

def get_mask_data_in_roi(mask, roi, target_voxel_size):

    assert mask.voxel_size.is_multiple_of(target_voxel_size), (
        "Can not upsample from %s to %s" % (mask.voxel_size, target_voxel_size))

    aligned_roi = roi.snap_to_grid(mask.voxel_size, mode='grow')
    aligned_data = mask.to_ndarray(aligned_roi, fill_value=0)

    if mask.voxel_size == target_voxel_size:
        return aligned_data

    factor = mask.voxel_size/target_voxel_size

    upsampled_aligned_data = upsample(aligned_data, factor)

    upsampled_aligned_mask = daisy.Array(
        upsampled_aligned_data,
        roi=aligned_roi,
        voxel_size=target_voxel_size)

    return upsampled_aligned_mask.to_ndarray(roi)

def mask_in_block(
        block,
        seg,
        masked_ds,
        mask):

    seg_data = seg.intersect(block.read_roi)
    seg_data.materialize()

    mask_data = get_mask_data_in_roi(mask, seg_data.roi, seg_data.voxel_size)

    seg_data = seg_data.to_ndarray(block.read_roi)

    print('Masking in block %s' %block.read_roi)
    seg_data *= mask_data.astype(np.uint64)

    masked_ds[block.write_roi] = seg_data

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    mask_neurons(**config)










