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

phase_switch = 10000

with open('train_net_config.json', 'r') as f:
    affs_config = json.load(f)

def add_malis_loss(graph):
    affs = graph.get_tensor_by_name(affs_config['affs'])
    gt_affs = graph.get_tensor_by_name(affs_config['gt_affs'])
    gt_seg = tf.placeholder(tf.int64, shape=(48, 56, 56), name='gt_seg')
    gt_affs_mask = graph.get_tensor_by_name(affs_config['affs_loss_weights'])

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
    print("Training in phase %s until %i"%(phase, max_iteration))

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

    with open('sd_net_config.json', 'r') as f:
        sd_config = json.load(f)
    
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

    train_inputs = {
        sd_config['raw']: raw,
        affs_config['embedding']: embedding,
        affs_config['gt_affs']: gt,
        affs_config['affs_loss_weights']: gt_scale,
    }
    if phase == 'euclid':
        train_loss = affs_config['loss']
        train_optimizer = affs_config['optimizer']
    else:
        train_loss = None
        train_optimizer = add_malis_loss
        train_inputs['gt_seg:0'] = labels


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
            checkpoint='../setup02/train_net_checkpoint_400000',
            graph='sd_net.meta',
            inputs={
                sd_config['raw']: raw
            },
            outputs={
                sd_config['embedding']: embedding
            }) +
        Train(
            'train_net',
            optimizer=train_optimizer,
            loss=train_loss,
            inputs=train_inputs,
            outputs={
                affs_config['affs']: affs
            },
            gradients={
                affs_config['affs']: affs_gradient
            },
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                embedding: 'volumes/embedding',
                labels: 'volumes/labels/neuron_ids',
                gt: 'volumes/labels/gt_affinities',
                affs: 'volumes/labels/pred_affinities',
            },
            dataset_dtypes={
                labels: np.uint64
            },
            every=100,
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
