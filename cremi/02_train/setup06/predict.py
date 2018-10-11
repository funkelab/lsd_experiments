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
    with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    embedding = ArrayKey('EMBEDDING')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size
    read_roi *= voxel_size
    write_roi *= voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(embedding, output_size)

    pipeline = (
        N5Source(
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
                config['embedding']: embedding
            },
            # TODO: change to predict graph
            graph=os.path.join(setup_dir, 'config.meta')
        ) +
        N5Write(
            dataset_names={
                embedding: 'volumes/lsds',
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

    out_file = run_config['out_file']
    out_dataset = run_config['out_dataset']

    '''f = z5py.File(out_file, use_zarr_format=False, mode='r+')
    if out_dataset not in f:
        ds = f.create_dataset(
            out_dataset,
            shape=(10,) + (write_roi//voxel_size).get_shape(),
            chunks=(10,) + output_shape,
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
