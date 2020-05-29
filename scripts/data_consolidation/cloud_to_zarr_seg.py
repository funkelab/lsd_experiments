import cloudvolume
# import daisy
import json
import logging
import numpy as np
import os
import sys

logging.basicConfig(level=logging.INFO)

def convert_coords(
        coords,
        voxel_size,
        method='world_to_vox',
        flip_coords=False):

    if method=='world_to_vox':
        if flip_coords:
            return [int(i/j) for i,j in zip(coords, voxel_size)][::-1]
        else:
            return [int(i/j) for i,j in zip(coords, voxel_size)]
    else:
        if flip_coords:
            return [int(i*j) for i,j in zip(coords, voxel_size)][::-1]
        else:
            return [int(i*j) for i,j in zip(coords, voxel_size)]

def fetch_in_block(
        block,
        voxel_size,
        seg_data,
        out_ds):

    logging.info('Fetching seg in block %s' %block.read_roi)

    voxel_size = list(voxel_size)

    block_start = list(block.write_roi.get_begin())
    block_end = list(block.write_roi.get_end())

    block_start = convert_coords(block_start,voxel_size)
    block_end = convert_coords(block_end,voxel_size)

    z_start, z_end = block_start[0], block_end[0]
    y_start, y_end = block_start[1], block_end[1]
    x_start, x_end = block_start[2], block_end[2]

    seg = seg_data[x_start:x_end, y_start:y_end, z_start:z_end]

    seg = np.array(np.transpose(seg[...,0],[2,1,0]))

    out_ds[block.write_roi] = seg

def get_cloud_roi(cloud_vol):

    info = cloud_vol.info

    print(info)

    scale_0 = info['scales'][0]

    roi_offset = scale_0['voxel_offset']
    roi_shape = scale_0['size']

    return roi_offset, roi_shape

def fetch(
        in_vol,
        voxel_size,
        roi_offset,
        roi_shape,
        out_file,
        out_ds,
        num_workers):

    total_roi = daisy.Roi((roi_offset), (roi_shape))

    read_roi = daisy.Roi((0, 0, 0), (3600, 3600, 3600))
    write_roi = read_roi

    logging.info('Creating out dataset...')

    seg_out = daisy.prepare_ds(
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
                in_vol,
                seg_out),
            fit='shrink',
            num_workers=num_workers) 
if __name__ == '__main__':

    in_vol = "https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/ffn_segmentation"

    seg_vol = cloudvolume.CloudVolume(
            in_vol,
            bounded=True,
            progress=True,
            fill_missing=True)

    info = seg_vol.info

    # print(info)

    for scale in info['scales']:
        scale['voxel_offset'] = [0, 0, 0]

    seg_vol.info = info

    print(seg_vol.info)

    roi_offset, roi_shape = get_cloud_roi(seg_vol)

    roi_shape = convert_coords(
                    roi_shape,
                    [10,10,20],
                    method='vox_to_world',
                    flip_coords=True)

    # voxel_size = daisy.Coordinate((20,10,10))

    print(roi_offset, roi_shape)

  #   out_file = sys.argv[1]
    # out_ds = 'volumes/ffn_segmentation_10_10_20'

    # fetch(
        # seg_vol,
        # voxel_size,
        # roi_offset,
        # roi_shape,
        # out_file,
        # out_ds,
        # num_workers=32)

