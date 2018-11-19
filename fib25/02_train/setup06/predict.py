from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import numpy as np
import os
import sys
import logging

def predict(iteration, in_file, read_roi, out_file):

    setup_dir = os.path.dirname(os.path.realpath(__file__))

    # TODO: change to predict graph
    with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    affs = ArrayKey('AFFS')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        ZarrSource(
            in_file,
            datasets = {
                raw: 'volumes/raw'
            },
        ) +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            inputs={
                config['raw']: raw
            },
            outputs={
                config['affs']: affs
            },
            graph=os.path.join(setup_dir, 'test_net.meta')
        ) +
        IntensityScaleShift(affs, 255, 0) +
        ZarrWrite(
            dataset_names={
                affs: 'volumes/affs',
            },
            output_filename=out_file
        ) +
        PrintProfilingStats(every=10) +
        Scan(chunk_request)
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
    
    predict(
        config['iteration'],
        config['raw_file'],
        read_roi,
        config['out_file'])
