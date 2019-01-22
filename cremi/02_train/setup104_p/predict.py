from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys

setup_dir = os.path.dirname(os.path.realpath(__file__))

print('setup directory:', setup_dir)

with open(os.path.join(setup_dir, 'test_net.json'), 'r') as f:
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
        lsds_file,
        lsds_dataset
        read_roi,
        out_file,
        out_dataset):

    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    chunk_request = BatchRequest()

    chunk_request.add(lsds, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        ZarrSource(
            lsds_file,
            datasets = {
                lsds: lsds_dataset
            }
        ) +
        Pad(lsds, size=None) +
        Crop(lsds, read_roi) +
        Predict(
            checkpoint=os.path.join(
                setup_dir,
                'train_net_checkpoint_%d'%iteration),
            graph=os.path.join(setup_dir, 'test_net.meta'),
            inputs={
                net_config['embedding']: lsds
            },
            outputs={
                net_config['affs']: affs
            }
        ) +
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
    logging.getLogger('gunpowder.nodes.n5_write').setLevel(logging.DEBUG)
    logging.getLogger('gunpowder.nodes.n5_source').setLevel(logging.DEBUG)

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
        run_config['lsds_file'],
        run_config['lsds_dataset'],
        read_roi,
        run_config['out_file'],
        run_config['out_dataset'])
