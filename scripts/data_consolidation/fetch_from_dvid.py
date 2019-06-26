import daisy
import logging
import zarr
import sys
import numpy as np
import os
import time
from neuclease.dvid import fetch_raw, fetch_labelmap_voxels, fetch_roi

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def scale_coords(
        in_array
        voxel_size,
        mode='v_to_n'):

    return [int(i*j) if mode == 'v_to_n' else int(i/j) for i,j in zip(in_array, voxel_size)]

def fetch_block(
        block,
        raw_in,
        raw_out,
        seg_in,
        seg_out,
        voxel_size):

    block_start = scale_coords(list(block.read_roi.get_begin()), voxel_size, mode='n_to_v')
    block_end = scale_coords(list(block.read_roi.get_end()), voxel_size, mode='n_to_v')

    block_bb = [block_start, block_end]

    raw_data = fetch_raw(*raw_in, block_bb)
    seg_data = fetch_labelmap_voxels(*seg_in, block_bb)

    raw_out[block.write_roi] = raw_data
    seg_out[block.write_roi] = seg_data

def fetch_volume(
        out_file,
        raw_in,
        seg_in,
        bb_start,
        bb_size,
        voxel_size):

    bb_start = daisy.Coordinate(bb_start)
    bb_size = daisy.Coordinate(bb_size)

    total_roi = daisy.Roi((bb_start), (bb_size))

    print(total_roi)

    read_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))
    write_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))

    raw_out = daisy.prepare_ds(
            out_file,
            "volumes/raw",
            total_roi,
            voxel_size=[8,8,8],
            dtype="uint8",
            write_roi=write_roi)

    seg_out = daisy.prepare_ds(
            out_file,
            "volumes/labels/neuron_ids",
            total_roi,
            voxel_size=[8,8,8],
            dtype="uint64",
            write_roi=write_roi)

    daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: fetch_block(
                b,
                raw_in,
                raw_out,
                seg_in,
                seg_out,
                voxel_size),
            fit='shrink',
            num_workers=24,
            read_write_conflict=False)

if __name__ == '__main__':

    raw_in = (#dvid uuid, #port, #volume)
    seg_in = (#dvid uuid, #port, #volume)

    voxel_size = [8, 8, 8] # or whatever

    bb_start = [17400, 25440, 24600] # voxels

    bb_start = scale_coords(bb_start, voxel_size) #nm

    bb_size = [20000, 20000, 20000]

    out_file = #path to out file

    fetch_volume(
            out_file,
            raw_in,
            seg_in,
            bb_start,
            bb_size,
            voxel_size)

    ## steps to fetch neuropil mask --> make optional
    ## fetches at scale 5 resolution

    # mask, mask_box = fetch_roi(dvid uuid, port, volume)
    # mask_box = (2**5) * mask_box

    # mask = mask.astype(np.uint8)

    # offset_world = [int(i*j) for i, j in zip(voxel_size, mask_box[0])]

    # f_out = zarr.open(out_file, mode='r+')

    # ds_out = f_out.create_dataset(
            # '/volumes/labels/mask',
            # data=mask,
            # compressor=zarr.get_codec({'id': 'gzip', 'level': 5}))

    # ds_out.attrs['resolution'] = [256, 256, 256]
    # ds_out.attrs['offset'] = offset_world






