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
import logging

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data/glia_mask/'
artifacts_dir = '../../01_data/training/'

samples = [
    'sample_0',
    'sample_1',
    'sample_2',
    'sample_A',
    'sample_B',
    'sample_C',
    'sample_A+',
    'sample_B+',
    'sample_C+'
]

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

experiment_dir = os.path.join(setup_dir, '..', '..')
affs_setup_dir = os.path.realpath(os.path.join(
    experiment_dir,
    '02_train',
    config['affs_setup']))

affinity_neighborhood = np.array([
    
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


def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    raw = ArrayKey('RAW')
    raw_cropped = ArrayKey('RAW_CROPPED')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    artifacts = ArrayKey('ARTIFACTS')
    artifacts_mask = ArrayKey('ARTIFACTS_MASK')
    pretrained_affs = ArrayKey('PRETRAINED_AFFS')
    affs = ArrayKey('PREDICTED_AFFS')
    gt = ArrayKey('GT_AFFINITIES')
    gt_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_scale = ArrayKey('GT_AFFINITIES_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    with open('train_affs_net.json', 'r') as f:
        affs_1_config = json.load(f)
    with open('train_net.json', 'r') as f:
        affs_2_config = json.load(f)

    voxel_size = Coordinate((40, 4, 4))
    affs_1_input_size = Coordinate(affs_1_config['input_shape'])*voxel_size
    affs_2_input_size = Coordinate(affs_2_config['input_shape'])*voxel_size
    pretrained_affs_size = Coordinate(affs_2_config['input_shape'])*voxel_size
    output_size = Coordinate(affs_2_config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, affs_1_input_size)
    request.add(raw_cropped, affs_2_input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(pretrained_affs, pretrained_affs_size)
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
                raw_cropped: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_mask: 'volumes/labels/mask',
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                raw_cropped: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Normalize(raw_cropped) +
        Pad(raw, None) +
        Pad(raw_cropped, None) +
        RandomLocation() +
        Reject(mask=labels_mask)
        for sample in samples
    )

    artifact_source = (
        Hdf5Source(
            os.path.join(artifacts_dir, 'sample_ABC_padded_20160501.defects.hdf'),
            datasets = {
                artifacts: 'defect_sections/raw',
                artifacts_mask: 'defect_sections/mask',
            },
            array_specs = {
                artifacts: ArraySpec(
                    voxel_size=(40, 4, 4),
                    interpolatable=True),
                artifacts_mask: ArraySpec(
                    voxel_size=(40, 4, 4),
                    interpolatable=True),
            }
        ) +
        RandomLocation(min_masked=0.05, mask=artifacts_mask) +
        Normalize(artifacts) +
        IntensityAugment(artifacts, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2])
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        IntensityAugment(raw_cropped, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True) +
        AddAffinities(
            affinity_neighborhood, 
            labels=labels,
            affinities=gt,
            labels_mask=labels_mask,
            affinities_mask=gt_mask) +
        BalanceLabels(
            gt,
            gt_scale,
            gt_mask) +
        DefectAugment(
            raw,
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            artifacts=artifacts,
            artifacts_mask=artifacts_mask,
            contrast_scale=0.5,
            axis=0) +
        DefectAugment(
            raw_cropped,
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            artifacts=artifacts,
            artifacts_mask=artifacts_mask,
            contrast_scale=0.5,
            axis=0) +
        IntensityScaleShift(raw, 2,-1) +
        IntensityScaleShift(raw_cropped, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Predict(
            checkpoint='../setup64_p/train_net_checkpoint_400000',
            graph='train_affs_net.meta',
            inputs={
                affs_1_config['raw']: raw
            },
            outputs={
                affs_1_config['affs']: pretrained_affs
            }) +
        Train(
            'train_net',
            optimizer=affs_2_config['optimizer'],
            loss=affs_2_config['loss'],
            inputs={
                affs_2_config['raw']: raw_cropped,
                affs_2_config['pretrained_affs']: pretrained_affs,
                affs_2_config['gt_affs']: gt,
                affs_2_config['affs_loss_weights']: gt_scale,
            },
            outputs={
                affs_2_config['affs']: affs
            },
            gradients={
                affs_2_config['affs']: affs_gradient
            },
            summary=affs_2_config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        IntensityScaleShift(raw_cropped, 0.5, 0.5) +
        Snapshot({
                raw_cropped: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt: 'volumes/labels/gt_affinities',
                affs: 'volumes/labels/pred_affinities',
                labels_mask: 'volumes/labels/mask',
                affs_gradient: 'volumes/affs_gradient'
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
