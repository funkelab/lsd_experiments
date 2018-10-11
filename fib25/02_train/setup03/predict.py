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
        in_file,
        raw_dataset,
        read_roi,
        out_file,
        out_dataset,
        write_roi):

    setup_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(setup_dir, 'test_net_config.json'), 'r') as f:
        aff_net_config = json.load(f)
    with open(os.path.join(setup_dir, 'lsd_net_config.json'), 'r') as f:
        lsd_net_config = json.load(f)
    experiment_dir = os.path.join(setup_dir, '..', '..')
    lsd_setup_dir = os.path.realpath(os.path.join(
        experiment_dir,
        '02_train',
        aff_net_config['lsd_setup']))


    raw = ArrayKey('RAW') 
    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(lsd_net_config['input_shape'])*voxel_size
    lsd_size = Coordinate(lsd_net_config['output_shape'])*voxel_size
    assert lsd_size == Coordinate(aff_net_config['input_shape'])*voxel_size
    output_size = Coordinate(aff_net_config['output_shape'])*voxel_size
    read_roi *= voxel_size
    write_roi *= voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, lsd_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        N5Source(
            in_file,
            datasets = {
                raw: raw_dataset
            },
        ) +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(lsd_setup_dir,
                'train_net_checkpoint_%d'%aff_net_config['lsd_iteration']),
            graph=os.path.join(setup_dir, 'lsd_net.meta'),
            inputs={
                lsd_net_config['raw']: raw
            },
            outputs={
                lsd_net_config['embedding']: lsds
            }
        ) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            inputs={
                aff_net_config['embedding']: lsds
            },
            outputs={
                aff_net_config['affs']: affs
            },
            graph=os.path.join(setup_dir, 'test_net.meta')
        ) +
        N5Write(
            dataset_names={
                affs: out_dataset,
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
        config['read_size'])
    write_roi = Roi(
        config['write_begin'],
        config['write_size'])
    
    if 'raw_dataset' in config:
        raw_dataset = config['raw_dataset']
    else:
        raw_dataset = 'volumes/raw'

    predict(
        config['iteration'],
        config['in_file'],
        raw_dataset,
        read_roi,
        config['out_file'],
        config['out_dataset'],
        write_roi)
