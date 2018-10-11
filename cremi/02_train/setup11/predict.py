from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
import z5py
import h5py

setup_dir = os.path.dirname(os.path.realpath(__file__))

print('setup directory:', setup_dir)

with open(os.path.join(setup_dir, 'affs_net_config.json'), 'r') as f:
    aff_net_config = json.load(f)
with open(os.path.join(setup_dir, 'lsd_net_config.json'), 'r') as f:
    lsd_net_config = json.load(f)

experiment_dir = os.path.join(setup_dir, '..', '..')
lsd_setup_dir = os.path.realpath(os.path.join(
    experiment_dir,
    '02_train',
    aff_net_config['lsd_setup']))

# voxels
input_shape = Coordinate(lsd_net_config['input_shape'])
lsds_shape = Coordinate(lsd_net_config['output_shape'])
assert lsds_shape == Coordinate(aff_net_config['input_shape'])
output_shape = Coordinate(aff_net_config['output_shape'])

context = (input_shape - output_shape)//2
print("Context is %s"%(context,))

# nm
voxel_size = Coordinate((40, 4, 4))
context_nm = context*voxel_size
input_size = input_shape*voxel_size
lsds_size = lsds_shape*voxel_size
output_size = output_shape*voxel_size

def predict(
        iteration,
        in_file,
        raw_dataset,
        read_roi,
        out_file,
        out_dataset):


    raw = ArrayKey('RAW')
    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    chunk_request = BatchRequest()

    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, lsds_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        N5Source(
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
            checkpoint=os.path.join(
                lsd_setup_dir,
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
            checkpoint=os.path.join(
                setup_dir,
                'train_net_checkpoint_%d'%iteration),
            inputs={
                aff_net_config['embedding']: lsds
            },
            outputs={
                aff_net_config['affs']: affs
            }
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
    #write_roi = read_roi.grow(-context_nm, -context_nm)

    print("Read ROI in nm is %s"%read_roi)
    print("Write ROI in nm is %s"%write_roi)

    '''f = z5py.File(out_file, use_zarr_format=False, mode='w')
    if out_dataset not in f:
        ds = f.create_dataset(
            out_dataset,
            shape=(3,) + (write_roi//voxel_size).get_shape(),
            chunks=(3,) + output_shape,
            compression='gzip',
            dtype=np.float32)
        ds.attrs['resolution'] = voxel_size[::-1]
        ds.attrs['offset'] = write_roi.get_begin()[::-1]'''

    if 'raw_dataset' in run_config:
        raw_dataset = run_config['raw_dataset']
    else:
        raw_dataset = 'volumes/raw'

    predict(
        run_config['iteration'],
        run_config['in_file'],
        raw_dataset,
        read_roi,
        run_config['out_file'],
        run_config['out_dataset'])
