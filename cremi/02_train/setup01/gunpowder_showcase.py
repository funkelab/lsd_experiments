from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import os
import math
import json
import numpy as np
import random

data_dir = '../../01_data/training'

# Used to create repeatedly the same augmentations, even when requests come from
# different parts of the pipeline
class SetSeed(BatchFilter):
    def prepare(self, request):
        random.seed(42)
    def process(self, batch, request):
        pass

def showcase():

    with open('train_net_config.json', 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    affs = ArrayKey('PREDICTED_AFFS')
    gt = ArrayKey('GT_AFFINITIES')
    gt_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_scale = ArrayKey('GT_AFFINITIES_SCALE')

    voxel_size = Coordinate((8,8,8))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)

    pipeline = (
        Hdf5Source(
            os.path.join(data_dir, 'cube01.hdf'),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/neuron_ids_mask',
            },
        ) +
        Normalize(raw) +
        Pad(raw, None)
    )

    with build(pipeline + Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=1,
            output_dir='showcase',
            output_filename='01.hdf')) as p:
        p.request_batch(request)

    pipeline = (
        pipeline +
        RandomLocation() +
        SetSeed() +
        Reject(mask=labels_mask)
    )
    request.add(labels_mask, output_size)

    with build(pipeline + Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=1,
            output_dir='showcase',
            output_filename='02.hdf')) as p:
        p.request_batch(request)

    pipeline = (
        pipeline +
        ElasticAugment(
            [40,40,40],
            [2,2,2],
            [0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8) +
        SetSeed()
    )

    with build(pipeline + Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=1,
            output_dir='showcase',
            output_filename='03.hdf')) as p:
        p.request_batch(request)

    pipeline = (
        pipeline +
        GrowBoundary(labels, labels_mask, steps=1) +
        AddAffinities(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            labels=labels,
            affinities=gt,
            labels_mask=labels_mask,
            affinities_mask=gt_mask)
    )
    request.add(gt, output_size)
    request.add(gt_mask, output_size)

    with build(pipeline + Snapshot({
                gt: 'volumes/labels/gt_affinities',
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=1,
            output_dir='showcase',
            output_filename='04.hdf')) as p:
        p.request_batch(request)

    pipeline = (
        pipeline +
        BalanceLabels(
            gt,
            gt_scale,
            gt_mask) +
        IntensityScaleShift(raw, 2,-1) +
        Train(
            'train_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['gt_affs']: gt,
                config['affs_loss_weights']: gt_scale,
            },
            outputs={
                config['affs']: affs
            },
            gradients={}) +
        IntensityScaleShift(raw, 0.5, 0.5)
    )
    request.add(gt_scale, output_size)
    request.add(affs, output_size)

    with build(pipeline + Snapshot({
                affs: 'volumes/affinities'
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=1,
            output_dir='showcase',
            output_filename='05.hdf')) as p:
        p.request_batch(request)

if __name__ == "__main__":
    set_verbose(True)
    showcase()
