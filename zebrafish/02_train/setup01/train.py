from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
from mala.gunpowder import AddLocalShapeDescriptor
import malis
import os
import math
import json
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# logging.getLogger('gunpowder').setLevel(logging.DEBUG)

data_dir = '../../01_data/'

samples = ['cube_01.zarr']

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

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
    labels_mask = ArrayKey('GT_LABELS_MASK')
    unlabelled = ArrayKey('UNLABELLED')
    lsds = ArrayKey('PREDICTED_LSDS')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    gt_affs_mask = ArrayKey('GT_AFFS_MASK')
    gt_lsds = ArrayKey('GT_LSDS')
    gt_lsds_scale = ArrayKey('GT_LSDS_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')
    lsds_gradient = ArrayKey('LSDS_GRADIENT')

    input_shape = config['input_shape']
    output_shape = config['output_shape']

    voxel_size = Coordinate((12,12,12))
    input_size = Coordinate(input_shape)*voxel_size
    output_size = Coordinate(output_shape)*voxel_size

    p = int(round(np.sqrt(np.sum([i*i for i in output_shape]))/2))

    labels_padding = Coordinate(
                        [j * round(i/j) for i,j in zip((p,)*3,list(voxel_size))]
                    )

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_lsds, output_size)
    request.add(gt_lsds_scale, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_scale, output_size)
    request.add(gt_affs_mask, output_size)

    snapshot_request = BatchRequest({
        lsds: request[gt_lsds],
        affs: request[gt_affs],
        affs_gradient: request[gt_affs],
        lsds_gradient: request[gt_lsds]
    })

    data_sources = tuple(
        ZarrSource(
            os.path.join(data_dir, sample),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/labels_mask',
                unlabelled: 'volumes/labels/unlabelled',
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False),
                unlabelled: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        Pad(labels, None) +
        Pad(labels_mask, labels_padding) +
        Pad(unlabelled, labels_padding) + 
        RandomLocation(mask=unlabelled, min_masked=0.5)
        for sample in samples
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[40,40,40],
            jitter_sigma=[2,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8) +
        SimpleAugment() +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
        GrowBoundary(labels, labels_mask, steps=1)+
        AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            mask=gt_lsds_scale,
            sigma=120,
            downsample=2) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabelled,
            affinities_mask=gt_affs_mask) +
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=20) +
        Train(
            'train_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['gt_lsds']: gt_lsds,
                config['loss_weights_lsds']: gt_lsds_scale,
                config['gt_affs']: gt_affs,
                config['loss_weights_affs']: gt_affs_scale
            },
            outputs={
                config['lsds']: lsds,
                config['affs']: affs
            },
            gradients={
                config['lsds']: lsds_gradient,
                config['affs']: affs_gradient
            },
            summary=config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/labels_mask',
                unlabelled: 'volumes/labels/unlabelled',
                gt_lsds: 'volumes/gt_lsds',
                lsds: 'volumes/pred_lsds',
                gt_affs: 'volumes/gt_affs',
                affs: 'volumes/pred_affs',
                affs_gradient: 'volumes/affs_gradient',
                lsds_gradient: 'volumes/lsds_gradient'
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=1000,
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
