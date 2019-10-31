from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
import os
import glob
import math
import json
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data/mask_data'

samples = glob.glob(os.path.join(data_dir, '*.n5'))


def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net.json', 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    logits = ArrayKey('LOGITS')
    mask = ArrayKey('GT_LABELS_MASK')
    pred_labels = ArrayKey('PREDICTED_MASK')
    logits_gradient = ArrayKey('LOGITS_GRADIENTS')

    voxel_size = Coordinate((40, 36, 36))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    p = int(round(np.sqrt(np.sum([i*i for i in config['output_shape']]))/2))
    labels_padding = Coordinate([j * round(i/j) for i,j in zip([p,]*3, list(voxel_size))])

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(mask, output_size)

    snapshot_request = BatchRequest({
        pred_labels: request[labels],
        logits: request[labels],
        logits_gradient: request[labels]
    })

    data_sources = tuple(
        ZarrSource(
            sample,
            datasets = {
                raw: 'volumes/raw/s2',
                labels: 'volumes/labels/new_ids/s2',
                mask: 'volumes/labels/mask/s2',
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                mask: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) + 
        Pad(labels, None) +
        Pad(mask, labels_padding) +
        RandomLocation(min_masked=0.5, mask=mask)
        for sample in samples
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,4,10],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'train_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['labels']: labels,
                config['mask']: mask
            },
            outputs={
                config['pred_labels']: pred_labels,
                config['logits']: logits
            },
            gradients={
                config['logits']: logits_gradient
            },
            summary=config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/ids',
                mask: 'volumes/labels/mask',
                pred_labels: 'volumes/pred_mask',
                logits: 'volumes/logits',
                logits_gradient: 'volumes/logits_gradient'
            },
            dataset_dtypes={
                labels: np.uint8
            },
            every=100,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":

    iteration = int(sys.argv[1])
    train_until(iteration)
