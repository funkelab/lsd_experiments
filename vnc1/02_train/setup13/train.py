from __future__ import print_function
import os
import sys
import logging
from gunpowder import *
from gunpowder.tensorflow import *
from mala.gunpowder import AddLocalShapeDescriptor
import math
import json
import tensorflow as tf
import numpy as np
import glob

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data/'

samples = glob.glob(os.path.join(data_dir, '*.n5'))

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

experiment_dir = os.path.join(setup_dir, '..', '..')
auto_setup_dir = os.path.realpath(os.path.join(
    experiment_dir,
    '02_train',
    config['lsds_setup']))

with open('train_auto_net.json', 'r') as f:
    sd_config = json.load(f)
with open('train_net.json', 'r') as f:
    context_config = json.load(f)

neighborhood = np.array([

    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],

    [-2, 0, 0],
    [0, -3, 0],
    [0, 0, -3],

    [-3, 0, 0],
    [0, -9, 0],
    [0, 0, -9],

    [-4, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
])


class EnsureUInt8(BatchFilter):

    def __init__(self, array):
        self.array = array

    def prepare(self, request):
        pass

    def process(self, batch, request):
        batch[self.array].data = (batch[self.array].data*255.0).astype(np.uint8)


def flatten_samples(tensor):

    output_shape = tensor.get_shape().as_list()

    num_channels = output_shape[0]

    assert num_channels >= 2,\
            'Tensor to flatten must be at least 2D but given shape is %i'%num_channels

    assert len(output_shape) == 4, 'Tensor must be 4D but shape is %i'%len(output_shape)

    flattened = tf.reshape(tensor, (num_channels, -1))

    return flattened


def sorenson_dice_loss(
        input_tensor,
        target_tensor,
        gt_affs_mask=None,
        eps=1e-6):

    assert input_tensor.shape == target_tensor.shape,\
            'Input tensor shape %i does not match target tensor shape %i'\
            %(input_tensor.shape, target_tensor.shape)

    #invert
    input_tensor = 1 - input_tensor
    target_tensor = 1 - target_tensor

    #multiply
    if gt_affs_mask is not None:
        assert input_tensor.shape == gt_affs_mask.shape,\
                'Input tensor shape %i or target tensor shape %i\
                    does not match mask tensor shape %i'\
                    %(input_tensor.shape, target_tensor.shape, gt_affs_mask.shape)

        input_tensor *= tf.cast(gt_affs_mask, tf.float32)
        target_tensor *= tf.cast(gt_affs_mask, tf.float32)

    #flatten
    input_tensor = flatten_samples(input_tensor)
    target_tensor = flatten_samples(target_tensor)

    numerator = tf.reduce_sum((input_tensor * target_tensor), -1)

    input_prod = (input_tensor * input_tensor)
    target_prod = (target_tensor * target_tensor)

    denominator = tf.reduce_sum(input_prod, -1) + tf.reduce_sum(target_prod, -1)
    channelwise_loss = -2 * (numerator / tf.maximum(denominator, eps))

    loss = tf.reduce_sum(channelwise_loss)

    return loss


def add_sorenson_loss(graph):

    affs = graph.get_tensor_by_name(context_config['affs'])
    gt_affs = graph.get_tensor_by_name(context_config['gt_affs'])
    gt_affs_mask = tf.placeholder(tf.int64, shape=(12,48,56,56), name='gt_affs_mask')

    loss = sorenson_dice_loss(affs, gt_affs, gt_affs_mask=gt_affs_mask)

    summary = tf.summary.scalar('setup13_sorenson_loss', loss)

    opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8,
            name='sorenson_optimizer')

    optimizer = opt.minimize(loss)

    return (loss, optimizer)


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
    unlabelled = ArrayKey('UNLABELLED')
    pretrained_lsd = ArrayKey('PRETRAINED_LSD')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFINITIES')
    gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    voxel_size = Coordinate((40,4,4))
    sd_input_size = Coordinate(sd_config['input_shape'])*voxel_size
    context_input_size = Coordinate(context_config['input_shape'])*voxel_size
    pretrained_lsd_size = Coordinate(context_config['input_shape'])*voxel_size
    output_size = Coordinate(context_config['output_shape'])*voxel_size
    context = output_size/2

    request = BatchRequest()
    request.add(raw, sd_input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(pretrained_lsd, pretrained_lsd_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt_affs],
        affs_gradient: request[gt_affs]
    })

    data_sources = tuple(
        ZarrSource(
            sample,
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/masked_ids',
                labels_mask: 'volumes/labels/mask',
                unlabelled: 'volumes/labels/ids_mask',
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
        RandomLocation(mask=unlabelled)
        for sample in samples
    )

    train_pipeline = data_sources
    train_pipeline += RandomProvider()
    train_pipeline += ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8)
    train_pipeline += SimpleAugment(transpose_only=[1, 2])
    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    train_pipeline += GrowBoundary(labels, labels_mask, steps=1, only_xy=True)
    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabelled,
            affinities_mask=gt_affs_mask)
    train_pipeline += IntensityScaleShift(raw, 2,-1)
    train_pipeline += PreCache(
            cache_size=40,
            num_workers=10)
    train_pipeline += Predict(
            checkpoint=os.path.join(
                auto_setup_dir,
                'train_net_checkpoint_%d'%config['lsds_iteration']),
            graph='train_auto_net.meta',
            inputs={
                sd_config['raw']: raw
            },
            outputs={
                sd_config['embedding']: pretrained_lsd
            })
    train_pipeline += EnsureUInt8(pretrained_lsd)
    train_pipeline += Normalize(pretrained_lsd)

    train_inputs = {
            context_config['pretrained_lsd']: pretrained_lsd,
            context_config['gt_affs']: gt_affs,
            'gt_affs_mask:0': gt_affs_mask
    }

    train_pipeline += Train(
            'train_net',
            optimizer=add_sorenson_loss,
            loss=None,
            inputs=train_inputs,
            outputs={
                context_config['affs']: affs
            },
            gradients={
                context_config['affs']: affs_gradient
            },
            summary='setup13_sorenson_loss:0',
            log_dir='log',
            save_every=10000)
    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)
    train_pipeline += Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/labels_mask',
                unlabelled: 'volumes/labels/unlabelled',
                pretrained_lsd: 'volumes/labels/pretrained_lsd',
                gt_affs: 'volumes/labels/gt_affinities',
                affs: 'volumes/labels/pred_affinities',
                affs_gradient: 'volumes/affs_gradient'
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=100,
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
