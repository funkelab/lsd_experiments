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
    'trvol-250-1',
    'trvol-250-2',
    'tstvol-520-1',
    'tstvol-520-2',
]

phase_switch = 10000

with open('train_net_config.json', 'r') as f:
    config = json.load(f)

def add_malis_loss(graph):
    affs = graph.get_tensor_by_name(config['affs'])
    gt_affs = graph.get_tensor_by_name(config['gt_affs'])
    gt_seg = tf.placeholder(tf.int64, shape=(92, 92, 92), name='gt_seg')
    gt_affs_mask = graph.get_tensor_by_name(config['affs_loss_weights'])

    loss = malis.malis_loss_op(affs,
            gt_affs,
            gt_seg,
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            gt_affs_mask)
    opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8,
            name='malis_optimizer')
    optimizer = opt.minimize(loss)

    return (loss, optimizer)

def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    if trained_until < phase_switch and max_iteration > phase_switch:
        train_until(phase_switch)

    phase = 'euclid' if max_iteration <= phase_switch else 'malis'
    print("[INFO] training in phase {0} until {1}".format(phase, max_iteration))

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    affs = ArrayKey('PREDICTED_AFFS')
    gt = ArrayKey('GT_AFFINITIES')
    gt_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_scale = ArrayKey('GT_AFFINITIES_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    voxel_size = Coordinate((8,8,8))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(gt_mask, output_size)
    request.add(gt_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt],
        affs_gradient: request[gt]
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/mask',
            },
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        RandomLocation() +
        Reject(mask=labels_mask)
        for sample in samples
    )

    train_inputs = {
            config['raw']: raw,
            config['gt_affs']: gt,
            config['affs_loss_weights']: gt_scale,
    }
    if phase == 'euclid':
        train_loss = config['loss']
        train_optimizer = config['optimizer']
    else:
        train_loss = None
        train_optimizer = add_malis_loss
        train_inputs['gt_seg:0'] = labels


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
        Train(
            'train_net',
            optimizer=train_optimizer,
            loss=train_loss,
            inputs=train_inputs,
            outputs={
                config['affs']: affs
            },
            gradients={
                config['affs']: affs_gradient
            },
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
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
