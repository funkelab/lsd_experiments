import sys
from gunpowder import *
from gunpowder.tensorflow import *
from lsd.gp import AddLocalShapeDescriptor
import os
import math
import json
import torch
import numpy as np
import logging

from funlib.learn.torch.models import UNet, ConvPass

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

data_dir = '../../../../01_data/updated_gt'

samples = [
            'neuron_test_1.n5',
            'neuron_test_6.n5',
            'neuron_test_8.n5',
            'neuron_control_1.n5',
            'neuron_control_8.n5',
            'updated_neuron_test_1.n5',
            'updated_neuron_test_2.n5',
            'updated_neuron_control_1.n5',
            'updated_neuron_control_2.n5',
            'updated_neuron_control_3.n5',
            'updated_neuron_control_large.n5',
            'updated_neuron_test_large.n5']

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def train(iterations):

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    # embedding = ArrayKey('PREDICTED_EMBEDDING')
    # affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    # gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    gt_affs_mask = ArrayKey('GT_AFFS_MASK')
    gt_embedding = ArrayKey('GT_EMBEDDING')

    input_shape = Coordinate((84,268,268))
    output_shape = Coordinate((48,56,56))

    voxel_size = Coordinate((93, 62, 62))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    #calculated max padding...
    labels_padding = Coordinate((6138,6386,6386))

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_embedding, output_size)
    request.add(gt_affs, output_size)
    # request.add(gt_affs_scale, output_size)
    request.add(gt_affs_mask, output_size)

    sources = tuple(
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

    pipeline = sources

    pipeline += RandomProvider()

   #  pipeline += ElasticAugment(
            # control_point_spacing=[4,4,6],
            # jitter_sigma=[0,1,1],
            # rotation_interval=[0,math.pi/2.0],
            # prob_slip=0.01,
            # prob_shift=0.01,
            # max_misalign=5,
            # subsample=8)

    # pipeline += SimpleAugment(transpose_only=[1, 2])

    # pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)

    # pipeline += GrowBoundary(labels, labels_mask, steps=1, only_xy=True)

    pipeline += AddLocalShapeDescriptor(
            labels,
            gt_embedding,
            sigma=1302,
            downsample=2)

    pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            affinities_mask=gt_affs_mask)

   #  pipeline += BalanceLabels(
            # gt_affs,
            # gt_affs_scale,
            # gt_affs_mask)

    # pipeline += IntensityScaleShift(raw, 2,-1)

    # pipeline += PreCache(cache_size=40, num_workers=10)

  #   pipeline += Train(
            # 'train_net',
            # optimizer=config['optimizer'],
            # loss=config['loss'],
            # inputs={
                # config['raw']: raw,
                # config['gt_embedding']: gt_embedding,
                # config['loss_weights_embedding']: labels_mask,
                # config['gt_affs']: gt_affs,
                # config['loss_weights_affs']: gt_affs_scale
            # },
            # outputs={
                # config['embedding']: embedding,
                # config['affs']: affs
            # },
            # gradients={
                # config['embedding']: emb_gradient,
                # config['affs']: affs_gradient
            # },
            # summary=config['summary'],
            # log_dir='log',
            # save_every=10000)

    # pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                labels_mask: 'labels_mask',
                gt_embedding: 'gt_embedding',
                # pred_embedding: 'pred_embedding',
                gt_affs: 'gt_affs',
                # pred_affs: 'pred_affs',
            },
            every=1,
            output_filename='batch_{id}.zarr')

    pipeline += PrintProfilingStats(every=10)

    print("Starting training...")

    with build(pipeline) as b:
        for i in range(iterations):
            b.request_batch(request)

    print("Training finished")

if __name__ == "__main__":

    train(5)
