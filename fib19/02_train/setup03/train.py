from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
import malis
import os
import math
import json
import tensorflow as tf
import numpy as np

data_dir = '../../01_data/training'
samples = [
    'cube01',
    'cube02',
    'cube03',
]

def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    embedding = ArrayKey('EMBEDDING')
    affs = ArrayKey('PREDICTED_AFFS')
    gt = ArrayKey('GT_AFFINITIES')
    gt_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_scale = ArrayKey('GT_AFFINITIES_SCALE')

    with open('sd_net_config.json', 'r') as f:
        sd_config = json.load(f)
    with open('train_net_config.json', 'r') as f:
        affs_config = json.load(f)

    voxel_size = Coordinate((8,8,8))
    input_size = Coordinate(sd_config['input_shape'])*voxel_size
    embedding_size = Coordinate(affs_config['input_shape'])*voxel_size
    output_size = Coordinate(affs_config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(embedding, embedding_size)
    request.add(gt, output_size)
    request.add(gt_mask, output_size)
    request.add(gt_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt],
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/neuron_ids_mask',
            },
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        RandomLocation() +
        Reject(mask=labels_mask)
        for sample in samples
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            [40,40,40],
            [2,2,2],
            [0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8) +
        SimpleAugment() +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
        GrowBoundary(labels, labels_mask, steps=1) +
        AddAffinities(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            labels=labels,
            affinities=gt,
            labels_mask=labels_mask,
            affinities_mask=gt_mask) +
        BalanceLabels(
            gt,
            gt_scale,
            gt_mask) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Predict(
            checkpoint='../setup02/train_net_checkpoint_200000',
            graph='sd_net.meta',
            inputs={
                sd_config['raw']: raw
            },
            outputs={
                sd_config['embedding']: embedding
            }) +
        Train(
            'train_net',
            optimizer=affs_config['optimizer'],
            loss=affs_config['loss'],
            inputs={
                affs_config['embedding']: embedding,
                affs_config['gt_affs']: gt,
                affs_config['affs_loss_weights']: gt_scale,
            },
            outputs={
                affs_config['affs']: affs
            },
            gradients={},
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                embedding: 'volumes/embedding',
                labels: 'volumes/labels/neuron_ids',
                gt: 'volumes/labels/gt_affinities',
                affs: 'volumes/labels/pred_affinities',
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=10000,
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
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
