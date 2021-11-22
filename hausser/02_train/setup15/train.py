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

data_dir = '../../01_data/updated_gt'

samples = [
            'neuron_test_1.n5',
            'neuron_test_6.n5',
            'neuron_test_8.n5',
            'neuron_control_1.n5',
            'neuron_control_8.n5',
            'updated_neuron_control_1.n5']

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
    embedding = ArrayKey('PREDICTED_EMBEDDING')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    gt_affs_mask = ArrayKey('GT_AFFS_MASK')
    gt_embedding = ArrayKey('GT_EMBEDDING')
    affs_gradient = ArrayKey('AFFS_GRADIENT')
    emb_gradient = ArrayKey('EMB_GRADIENT')

    voxel_size = Coordinate((93, 62, 62))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    #calculated max padding...
    labels_padding = Coordinate((6138,6386,6386))

    print('Labels padding: ',labels_padding)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_embedding, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_scale, output_size)
    request.add(gt_affs_mask, output_size)

    snapshot_request = BatchRequest({
        embedding: request[gt_embedding],
        affs: request[gt_affs],
        affs_gradient: request[gt_affs],
        emb_gradient: request[gt_embedding]
    })

    data_sources = tuple(
        ZarrSource(
            os.path.join(data_dir, sample),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/filtered_ids',
                labels_mask: 'volumes/labels/labels_mask'
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        Pad(labels, labels_padding) +
        Pad(labels_mask, labels_padding) +
        RandomLocation(mask=labels_mask, min_masked=0.5)
        for sample in samples
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,4,6],
            jitter_sigma=[0,1,1],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=5,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True)+
        AddLocalShapeDescriptor(
            labels,
            gt_embedding,
            sigma=1302,
            downsample=2) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            affinities_mask=gt_affs_mask) +
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask) +
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
                config['gt_embedding']: gt_embedding,
                config['loss_weights_embedding']: labels_mask,
                config['gt_affs']: gt_affs,
                config['loss_weights_affs']: gt_affs_scale
            },
            outputs={
                config['embedding']: embedding,
                config['affs']: affs
            },
            gradients={
                config['embedding']: emb_gradient,
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
                gt_embedding: 'volumes/gt_embedding',
                embedding: 'volumes/pred_embedding',
                gt_affs: 'volumes/gt_affs',
                affs: 'volumes/pred_affs',
                affs_gradient: 'volumes/affs_gradient',
                emb_gradient: 'volumes/emb_gradient'
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
