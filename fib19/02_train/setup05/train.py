from __future__ import print_function
import os
import sys
import logging
from gunpowder import *
from gunpowder.tensorflow import *
from mala.gunpowder import AddLocalShapeDescriptor
import malis
import math
import json
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)

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

    with open('lsd_context_net_config.json', 'r') as f:
        context_config = json.load(f)
        
    with open('sd_net_config.json', 'r') as f:
        sd_config = json.load(f)

    raw = ArrayKey('RAW')
    raw_cropped = ArrayKey('RAW_CROPPED')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    pretrained_lsd = ArrayKey('PRETRAINED_LSD')
    embedding = ArrayKey('PREDICTED_EMBEDDING')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_embedding = ArrayKey('GT_EMBEDDING')
    gt_embedding_scale = ArrayKey('GT_EMBEDDING_SCALE')
    gt_affs = ArrayKey('GT_AFFINITIES')
    gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_affs_scale = ArrayKey('GT_AFFINITIES_SCALE')

    voxel_size = Coordinate((8,8,8))
    sd_input_size = Coordinate(sd_config['input_shape'])*voxel_size
    context_input_size = Coordinate(context_config['input_shape'])*voxel_size
    pretrained_lsd_size = Coordinate(context_config['input_shape'])*voxel_size
    output_size = Coordinate(context_config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, sd_input_size)
    request.add(raw_cropped, context_input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(pretrained_lsd, pretrained_lsd_size)
    request.add(gt_embedding, output_size)
    request.add(gt_embedding_scale, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        embedding: request[gt_embedding],
        affs: request[gt_affs],
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                raw: 'volumes/raw',
                raw_cropped: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/neuron_ids_mask',
            },
        ) +
        Normalize(raw) +
        Normalize(raw_cropped) +
        Pad(raw, None) +
        Pad(raw_cropped, None) +
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
        IntensityAugment(raw_cropped, 0.9, 1.1, -0.1, 0.1) +
        GrowBoundary(labels, labels_mask, steps=1) +
        AddLocalShapeDescriptor(
            labels,
            gt_embedding,
            mask=gt_embedding_scale,
            sigma=80,
            downsample=2) +
        AddAffinities(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            affinities_mask=gt_affs_mask) +
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask) +
        IntensityScaleShift(raw, 2,-1) +
        IntensityScaleShift(raw_cropped, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Predict(
            checkpoint='/groups/funke/home/funkej/workspace/projects/lsd/run/fib19/02_train/setup02/train_net_checkpoint_150000',
            graph='sd_net.meta',
            inputs={
                sd_config['raw']: raw
            },
            outputs={
                sd_config['embedding']: pretrained_lsd
            }) +
        Train(
            'lsd_context_net',
            optimizer=context_config['optimizer'],
            loss=context_config['loss'],
            inputs={
                context_config['raw']: raw_cropped,
                context_config['pretrained_lsd']: pretrained_lsd,
                context_config['gt_embedding']: gt_embedding,
                context_config['loss_weights_embedding']: gt_embedding_scale,
                context_config['gt_affs']: gt_affs,
                context_config['loss_weights_affs']: gt_affs_scale,
            },
            outputs={
                context_config['embedding']: embedding,
                context_config['affs']: affs
            },
            gradients={},
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        IntensityScaleShift(raw_cropped, 0.5, 0.5) +
        Snapshot({
                raw_cropped: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_embedding: 'volumes/labels/gt_embedding',
                embedding: 'volumes/labels/pred_embedding',
                gt_affs: 'volumes/labels/gt_affinities',
                affs: 'volumes/labels/pred_affinities',
            },
            dataset_dtypes={
                labels: np.uint64
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
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
