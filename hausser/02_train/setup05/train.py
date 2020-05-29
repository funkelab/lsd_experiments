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

data_dir = '../../01_data/'

samples = ['test_volume.zarr', 'control_volume.zarr']

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

class UnmaskBackground(BatchFilter):

    def __init__(self, target_mask, background_mask):
        self.target_mask = target_mask
        self.background_mask = background_mask

    def process(self, batch, request):
        batch[self.target_mask].data = np.logical_or(
                batch[self.target_mask].data,
                np.logical_not(batch[self.background_mask].data))

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
    embedding = ArrayKey('PREDICTED_EMBEDDING')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    gt_embedding = ArrayKey('GT_EMBEDDING')
    gt_embedding_scale = ArrayKey('GT_EMBEDDING_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')
    emb_gradient = ArrayKey('EMB_GRADIENT')

    voxel_size = Coordinate((93, 62, 62))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size
    context = output_size/2

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_embedding, output_size)
    request.add(gt_embedding_scale, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_scale, output_size)

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
                raw: 'train/raw_clahe/s0',
                labels: 'train/neuron_ids/s0',
                labels_mask: 'train/labels_mask/s0',
                unlabelled: 'train/unlabelled/s0',
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False),
                unlabelled: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(labels, context) +
        Pad(labels_mask, context) +
        Pad(unlabelled, context) + 
        RandomLocation(mask=unlabelled, min_masked=0.3)
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
            mask=gt_embedding_scale,
            sigma=1302,
            downsample=2) +
        UnmaskBackground(gt_embedding_scale, labels_mask) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            unlabelled=unlabelled) +
        BalanceLabels(
            gt_affs,
            gt_affs_scale) +
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
                config['loss_weights_embedding']: gt_embedding_scale,
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
                unlabelled: 'volumes/labels/unlabelled',
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
