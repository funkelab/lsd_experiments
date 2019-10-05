import cloudvolume
import daisy
import json
import logging
import numpy as np
import os
import sys

logging.basicConfig(level=logging.INFO)

def world_to_vox(offset,voxel_size):

    return [int(i/j) for i,j in zip(offset, voxel_size)]

def cloud_to_zarr_coords(coords, voxel_size):

    return [int(i*j) for i,j in zip(coords, voxel_size)][::-1]

def fetch_in_block(
        block,
        voxel_size,
        raw_data,
        out_ds):

    logging.info('Fetching raw in block %s' %block.read_roi)

    voxel_size = list(voxel_size)

    block_start = list(block.write_roi.get_begin())
    block_end = list(block.write_roi.get_end())

    block_start = world_to_vox(block_start,voxel_size)
    block_end = world_to_vox(block_end,voxel_size)

    z_start, z_end = block_start[0], block_end[0]
    y_start, y_end = block_start[1], block_end[1]
    x_start, x_end = block_start[2], block_end[2]

    raw = raw_data[x_start:x_end, y_start:y_end, z_start:z_end]

    raw = np.array(np.transpose(raw[...,0], [2,1,0]))

    out_ds[block.write_roi] = raw

def get_cloud_roi(cloud_vol):

    info = cloud_vol.info

    roi_offset = info.voxel_offset
    roi_shape = info['scales'][0].size

    return

def fetch(
        in_vol,
        voxel_size,
        out_file,
        out_ds,
        num_workers,
        roi_offset=None,
        roi_shape=None)

    raw_vol = cloudvolume.CloudVolume(
            in_vol,
            bounded=True,
            progress=True)

    info = raw_vol.info

    for scale in info['scales']:
        if not scale['voxel_offset']:
            scale['voxel_offset'] = [0,0,0]

    raw_vol.info = info

    total_roi = daisy.Roi((roi_offset), (roi_shape))

    read_roi = daisy.Roi((0, 0, 0), (3600, 3600, 3600))
    write_roi = read_roi

    logging.info('Creating out dataset...')

    raw_out = daisy.prepare_ds(
            out_file,
            out_ds,
            total_roi,
            voxel_size,
            dtype=np.uint64,
            write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: fetch_in_block(
                b,
                voxel_size,
                raw_vol,
                raw_out),
            fit='shrink',
            num_workers=num_workers)

if __name__ == '__main__':

    # in_vol = "https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned"

    in_vol = "https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/ffn_segmentation"

    raw_vol = cloudvolume.CloudVolume(
            in_vol,
            bounded=True,
            progress=True)

    info = raw_vol.info

    for scale in info['scales']:
        scale['voxel_offset'] = [0, 0, 0]

    raw_vol.info = info

    voxel_size = daisy.Coordinate((20,9,9))
    roi_offset = [0, 0, 0]
    roi_shape = [114000, 97920, 95616]

    out_file = sys.argv[1]
    out_ds = 'volumes/ffn_segmentation'

    fetch(
        raw_vol,
        voxel_size,
        roi_offset,
        roi_shape,
        out_file,
        out_ds,
        num_workers=32)

