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
samples = [
    'eb-inner-groundtruth-with-context-x20172-y2322-z14332.zarr',
    'eb-outer-groundtruth-with-context-x20532-y3512-z14332.zarr',
    'fb-inner-groundtruth-with-context-x17342-y4052-z14332.zarr',
    'fb-outer-groundtruth-with-context-x13542-y2462-z14332.zarr',
    'lh-groundtruth-with-context-x7737-y20781-z12444.zarr',
    'lobula-groundtruth-with-context-x3648-y12800-z29056.zarr',
    'pb-groundtruth-with-context-x8472-y2372-z9372.zarr',
    'pb-groundtruth-with-context-x8472-y2892-z9372.zarr'
]

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
    gt_embedding = ArrayKey('GT_EMBEDDING')
    gt_embedding_scale = ArrayKey('GT_EMBEDDING_SCALE')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    #max labels padding calculated
    labels_padding = Coordinate((608,768,768))

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_embedding, output_size)
    request.add(gt_embedding_scale, output_size)

    snapshot_request = BatchRequest({
        embedding: request[gt_embedding]
    })

    data_sources = tuple(
        ZarrSource(
            os.path.join(data_dir, sample),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/mask',
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
        RandomLocation(min_masked=0.5, mask=labels_mask)
        for sample in samples
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[40, 40, 40],
            jitter_sigma=[0, 0, 0],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=8) +
        SimpleAugment() +
        ElasticAugment(
            control_point_spacing=[40, 40, 40],
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
        GrowBoundary(labels, labels_mask, steps=1) +
        AddLocalShapeDescriptor(
            labels,
            gt_embedding,
            mask=gt_embedding_scale,
            sigma=80,
            downsample=2) +
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
                config['loss_weights_embedding']: gt_embedding_scale
            },
            outputs={
                config['embedding']: embedding
            },
            gradients={},
            summary=config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_embedding: 'volumes/gt_embedding',
                embedding: 'volumes/pred_embedding',
                labels_mask: 'volumes/labels/mask'
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
