import csv
import numpy as np
import os
import sys
import daisy
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def filter_neurons(
        whitelist_file,
        in_file,
        out_file,
        in_ds,
        out_ds,
        num_workers,
        header=False,
        ids_to_filter=None):

    '''Filter neurons in volume given a whitelist.

    Args:

        whitelist_file (``string``):

            Path of .csv file containing whitelisted neurons

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

        header (``bool``):

            Whether the csv contains a header, if true will be stripped before
            filtered

        ids_to_filter (``list of string(s)``):

            ID(s) in whitelist csv to filter segmentation by (i.e 'Traced'). If
            None, will filter segmentation by all ids in whitelist.

    '''

    #load csv
    whitelist = list(csv.reader(open(whitelist_file, 'rt'), delimiter=','))
    print('Loaded csv')

    #strip header if true
    if header:
        whitelist.pop(0)
        print('Stripped csv header')

    #remove proofreading status, filter if needed
    if ids_to_filter is not None:
        whitelist = [int(str.split(k[0])[0]) for k in whitelist for i in ids_to_filter if i in k[1]]
        print('Removed proofreading status and filtered whitelist by id: %s' %ids_to_filter)

    else:
        whitelist = [int(str.split(k[0])[0]) for k in whitelist]
        print('Removed proofreading status')

    print(whitelist)

    #load neuron dataset to filter
    seg = daisy.open_ds(in_file, in_ds)
    print('Loaded neuron ids')

    read_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))
    write_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))

    print('Read roi is %s, write roi is %s' %(read_roi, write_roi))

    #prepare the filtered id dataset
    print('Preparing out dataset...')
    filtered = daisy.prepare_ds(
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
            process_function=lambda b: filter_in_block(
                b,
                seg,
                filtered,
                whitelist),
            fit='shrink',
            num_workers=num_workers,
            read_write_conflict=False)

def filter_in_block(
        block,
        seg,
        filtered,
        whitelist):

    seg_data = seg.to_ndarray(block.write_roi)

    print('Filtering in block %s' %block.read_roi)
    seg_data[np.isin(seg_data, whitelist, invert=True)] = 0

    filtered[block.write_roi] = seg_data

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    filter_neurons(**config)










