from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import malis
import math
import numpy as np
import os
import sys
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data/training'
samples = [
    'trvol-250-1.zarr',
    'trvol-250-2.zarr',
    'tstvol-520-1.zarr',
    'tstvol-520-2.zarr',
]

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

# needs to match order of samples (small to large)
probabilities = [0.05, 0.05, 0.45, 0.45]

phase_switch = 10000

with open('train_net.json', 'r') as f:
        config = json.load(f)

def add_malis_loss(graph):
    affs = graph.get_tensor_by_name(config['affs'])
    gt_affs = graph.get_tensor_by_name(config['gt_affs'])
    gt_seg = tf.placeholder(tf.int64, shape=(92, 92, 92), name='gt_seg')

    loss = malis.malis_loss_op(affs, 
        gt_affs, 
        gt_seg,
        neighborhood)
    malis_summary = tf.summary.scalar('setup07_malis_loss', loss)
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
    print("Training in phase %s until %i"%(phase, max_iteration))

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    input_shape = config['input_shape']
    output_shape = config['output_shape']

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    #Assume worst case (rotation augmentation by 45 degrees) and pad
    #by half the length of the diagonal of the network output size

    p = int(round(np.sqrt(np.sum([i*i for i in output_shape]))/2))

    #Ensure that our padding is the closest multiple of our resolution

    labels_padding = Coordinate([j * round(i/j) for i,j in zip([p,p,p], list(voxel_size))])

    print('Labels padding:', labels_padding)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)

    if phase == 'euclid':
        request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt_affs],
        affs_gradient: request[gt_affs]
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

    train_pipeline = data_sources

    train_pipeline += RandomProvider(probabilities=probabilities)

    train_pipeline += ElasticAugment(
            control_point_spacing=[40, 40, 40],
            jitter_sigma=[0, 0, 0],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=8)

    train_pipeline += SimpleAugment()

    train_pipeline += ElasticAugment(
            control_point_spacing=[40,40,40],
            jitter_sigma=[2,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8)

    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    train_pipeline += GrowBoundary(labels, labels_mask, steps=1)

    if phase == 'malis':
        train_pipeline += RenumberConnectedComponents(
                labels=labels
                )

    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs)

    if phase == 'euclid':
        train_pipeline += BalanceLabels(
                gt_affs,
                gt_affs_scale)

    train_pipeline += IntensityScaleShift(raw, 2,-1) 

    train_pipeline += PreCache(
            cache_size=40,
            num_workers=10)

    train_inputs = {
            config['raw']: raw,
            config['gt_affs']: gt_affs
    }

    if phase == 'euclid':
        train_loss = config['loss']
        train_optimizer = config['optimizer']
        train_summary = config['summary']
        train_inputs[config['loss_weights_affs']] = gt_affs_scale
    else:
        train_loss = None
        train_optimizer = add_malis_loss
        train_inputs['gt_seg:0'] = labels
        train_summary = 'setup07_malis_loss:0'

    train_pipeline += Train(
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
            summary=train_summary,
            log_dir='log',
            save_every=10000)

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities',
                labels_mask: 'volumes/labels/mask',
                affs_gradient: 'volumes/affs_gradient'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_affs: np.float32
            },
            every=1000,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request)

    train_pipeline += PrintProfilingStats(every=10)

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":

    iteration = int(sys.argv[1])
    train_until(iteration)
