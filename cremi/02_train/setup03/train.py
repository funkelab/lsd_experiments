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
    'sample_A_padded_20160501.aligned.filled.cropped',
    'sample_B_padded_20160501.aligned.filled.cropped',
    'sample_C_padded_20160501.aligned.filled.cropped.0:90'
]

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'affs_net_config.json'), 'r') as f:
    aff_net_config = json.load(f)

experiment_dir = os.path.join(setup_dir, '..', '..')
lsd_setup_dir = os.path.realpath(os.path.join(
    experiment_dir,
    '02_train',
    aff_net_config['lsd_setup']))

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

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
    artifacts = ArrayKey('ARTIFACTS')
    artifacts_mask = ArrayKey('ARTIFACTS_MASK')
    embedding = ArrayKey('EMBEDDING')
    affs = ArrayKey('PREDICTED_AFFS')
    gt = ArrayKey('GT_AFFINITIES')
    gt_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_scale = ArrayKey('GT_AFFINITIES_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    with open('lsd_net_config.json', 'r') as f:
        sd_config = json.load(f)
    with open('train_net_config.json', 'r') as f:
        affs_config = json.load(f)

    voxel_size = Coordinate((40, 4, 4))
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
        affs_gradient: request[gt]
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids_notransparency',
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
        RandomLocation() +
        Reject(mask=labels_mask)
        for sample in samples
    )

    artifact_source = (
        Hdf5Source(
            os.path.join(data_dir, 'sample_ABC_padded_20160501.defects.hdf'),
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
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True) +
        AddAffinities(
            neighborhood,
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
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Predict(
            checkpoint=os.path.join(
                lsd_setup_dir,
                'train_net_checkpoint_%d'%aff_net_config['lsd_iteration']),
            graph='lsd_net.meta',
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
            gradients={
                affs_config['affs']: affs_gradient
            },
            summary=affs_config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                embedding: 'volumes/embedding',
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
    # set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
