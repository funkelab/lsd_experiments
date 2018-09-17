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
    'sample_A_padded_20160501.aligned.filled.cropped',
    'sample_B_padded_20160501.aligned.filled.cropped',
    'sample_C_padded_20160501.aligned.filled.cropped.0:90',
]

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net_config.json', 'r') as f:
        context_config = json.load(f)
        
    with open('sd_net_config.json', 'r') as f:
        sd_config = json.load(f)

    raw = ArrayKey('RAW')
    raw_cropped = ArrayKey('RAW_CROPPED')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    artifacts = ArrayKey('ARTIFACTS')
    artifacts_mask = ArrayKey('ARTIFACTS_MASK')
    pretrained_lsd = ArrayKey('PRETRAINED_LSD')
    embedding = ArrayKey('PREDICTED_EMBEDDING')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_embedding = ArrayKey('GT_EMBEDDING')
    gt_embedding_scale = ArrayKey('GT_EMBEDDING_SCALE')
    gt_affs = ArrayKey('GT_AFFINITIES')
    gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
    gt_affs_scale = ArrayKey('GT_AFFINITIES_SCALE')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    voxel_size = Coordinate((40,4,4))
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
        affs_gradient: request[gt_affs]
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                raw: 'volumes/raw',
                raw_cropped: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids_notransparency',
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
        IntensityAugment(raw_cropped, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True) +
        AddLocalShapeDescriptor(
            labels,
            gt_embedding,
            mask=gt_embedding_scale,
            sigma=80,
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
            checkpoint='../setup06/train_net_checkpoint_400000',
            graph='sd_net.meta',
            inputs={
                sd_config['raw']: raw
            },
            outputs={
                sd_config['embedding']: pretrained_lsd
            }) +
        Train(
            'train_net',
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
            gradients={
                context_config['affs']: affs_gradient
            },
            summary=context_config['summary'],
            log_dir='log',
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
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
