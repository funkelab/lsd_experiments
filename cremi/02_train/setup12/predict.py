from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    net_config = json.load(f)

# voxels
input_shape = Coordinate(net_config['input_shape'])
output_shape = Coordinate(net_config['output_shape'])
context = (input_shape - output_shape)//2
print("Context is %s"%(context,))

# nm
voxel_size = Coordinate((40, 4, 4))
context_nm = context*voxel_size
input_size = input_shape*voxel_size
output_size = output_shape*voxel_size

def predict(
        iteration,
        in_file,
        raw_dataset,
        read_roi,
        out_file,
        out_dataset):

    raw = ArrayKey('RAW')
    affs = ArrayKey('AFFS')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        ZarrSource(
            in_file,
            datasets = {
                raw: raw_dataset
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
            }
        ) +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            inputs={
                net_config['raw']: raw
            },
            outputs={
                net_config['affs']: affs
            },
            graph=os.path.join(setup_dir, 'config.meta')
        ) +
        IntensityScaleShift(affs, 255, 0) +
        ZarrWrite(
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
        run_config = json.load(f)

    read_roi = Roi(
        run_config['read_begin'],
        run_config['read_size'])
    write_roi = read_roi.grow(-context_nm, -context_nm)

    print("Read ROI in nm is %s"%read_roi)
    print("Write ROI in nm is %s"%write_roi)

    predict(
        run_config['iteration'],
        run_config['in_file'],
        run_config['in_dataset'],
        read_roi,
        run_config['out_file'],
        run_config['out_dataset'])
