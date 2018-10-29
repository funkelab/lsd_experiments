from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import numpy as np
import os
import sys
import logging

def predict(iteration, in_file, read_roi, out_file, out_dataset):

    setup_dir = os.path.dirname(os.path.realpath(__file__))

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
        N5Source(
            in_file,
            datasets = {
                raw: 'volumes/raw'
            },
            array_specs = {
                raw: ArraySpec(voxel_size=voxel_size)
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
        N5Write(
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

    print("Starting prediction...")

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)
    logging.getLogger('gunpowder.nodes.n5_write').setLevel(logging.DEBUG)
    logging.getLogger('gunpowder.nodes.n5_source').setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    read_roi = Roi(
        config['read_begin'],
        config['read_size'])

    predict(
        config['iteration'],
        config['in_file'],
        read_roi,
        config['out_file'],
        config['out_dataset'])
