from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import numpy as np
import os
import sys
import logging

def predict(
        iteration,
        lsds_file,
        lsds_dataset,
        read_roi,
        out_file,
        out_dataset):

    setup_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
        aff_net_config = json.load(f)
    experiment_dir = os.path.join(setup_dir, '..', '..')

    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(aff_net_config['input_shape'])*voxel_size
    output_size = Coordinate(aff_net_config['output_shape'])*voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(lsds, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        ZarrSource(
            lsds_file,
            datasets = {
                lsds: lsds_dataset
            },
        ) +
        Pad(lsds, size=None) +
        Crop(lsds, read_roi) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            inputs={
                aff_net_config['embedding']: lsds
            },
            outputs={
                aff_net_config['affs']: affs
            }
        ) +
        IntensityScaleShift(affs, 255, 0) +
        ZarrWrite(
            dataset_names={
                affs: out_dataset,
            },
            output_filename=out_file
        ) +
        PrintProfilingStats(every=10) +
        Scan(chunk_request, num_workers=1)
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
        config['read_size'])
    
    if 'lsds_dataset' in config:
        lsds_dataset = config['lsds_dataset']
    else:
        lsds_dataset = 'volumes/lsds'

    predict(
        config['iteration'],
        config['lsds_file'],
        lsds_dataset,
        read_roi,
        config['out_file'],
        config['out_dataset'])
