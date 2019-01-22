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
        raw_file,
        raw_dataset,
        lsds_file,
        lsds_dataset,
        read_roi,
        out_file,
        out_dataset):

    setup_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
        aff_net_config = json.load(f)
    experiment_dir = os.path.join(setup_dir, '..', '..')

    raw = ArrayKey('RAW')
    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(aff_net_config['input_shape'])*voxel_size
    output_size = Coordinate(aff_net_config['output_shape'])*voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        (ZarrSource(
            raw_file,
            datasets = {
                raw: raw_dataset
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True)
            }) +
         Pad(raw, size=None) +
         Crop(raw, read_roi) +
         Normalize(raw) +
         IntensityScaleShift(raw, 2, -1),

         ZarrSource(
             lsds_file,
             datasets = {
                 lsds: lsds_dataset
             }) +
         Pad(lsds, size=None) +
         Crop(lsds, read_roi)) +
        MergeProvider() +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            graph=os.path.join(setup_dir, 'test_net.meta')
            inputs={
                aff_net_config['raw']: raw,
                aff_net_config['pretrained_lsd']: lsds
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
    
    if 'lsds_dataset' in config:
        lsds_dataset = config['lsds_dataset']
    else:
        lsds_dataset = 'volumes/lsds'
    
    if 'raw_dataset' in config:
        raw_dataset = config['raw_dataset']
    else:
        raw_dataset = 'volumes/raw'

    predict(
        config['iteration'],
        config['raw_file']
        raw_dataset,
        config['lsds_file'],
        lsds_dataset,
        read_roi,
        config['out_file'],
        config['out_dataset'])
