from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import numpy as np
import os
import sys
import logging

def predict(iteration, in_file, read_roi, out_file, write_roi):

    setup_dir = os.path.dirname(os.path.realpath(__file__))

    # TODO: change to predict graph
    with open(os.path.join(setup_dir, 'train_net_config.json'), 'r') as f:
        config = json.load(f)

    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size
    read_roi *= voxel_size
    write_roi *= voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(lsds, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        N5Source(
            in_file,
            datasets = {
                lsds: 'volumes/lsds'
            },
        ) +
        Pad(lsds, size=None) +
        Crop(lsds, read_roi) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            inputs={
                config['embedding']: lsds
            },
            outputs={
                config['affs']: affs
            },
            # TODO: change to predict graph
            graph=os.path.join(setup_dir, 'train_net.meta')
        ) +
        N5Write(
            dataset_names={
                affs: 'volumes/affs_from_lsds',
            },
            output_filename=out_file
        ) +
        PrintProfilingStats(every=10) +
        Scan(chunk_request, num_workers=10)
    )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    read_roi = Roi(
        config['read_begin'],
        config['read_shape'])
    write_roi = Roi(
        config['write_begin'],
        config['write_shape'])

    predict(
        config['iteration'],
        config['in_file'],
        read_roi,
        config['out_file'],
        write_roi)
