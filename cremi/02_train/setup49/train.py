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

phase_switch = 10000

with open('train_net_config.json', 'r') as f:
    context_config = json.load(f)

def add_malis_loss(graph):
    affs = graph.get_tensor_by_name(context_config['affs'])
    gt_affs = graph.get_tensor_by_name(context_config['gt_affs'])
    gt_seg = tf.placeholder(tf.int64, shape=(48, 56, 56), name='gt_seg')
    gt_affs_mask = tf.placeholder(tf.int64, shape=(12,48,56,56), name='gt_affs_mask')

    loss = malis.malis_loss_op(affs, 
        gt_affs, 
        gt_seg,
        affinity_neighborhood,
        gt_affs_mask)
    malis_summary = tf.summary.scalar('setup49_malis_loss', loss)
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
    if phase is 'euclid':
        request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        embedding: request[gt_embedding],
        affs: request[gt_affs]
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

    if phase == 'malis':
        train_pipeline += RenumberConnectedComponents(
            labels=labels
            )

    train_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_embedding,
            mask=gt_embedding_scale,
            sigma=80,
            downsample=2)
    
    train_pipeline += AddAffinities(
            affinity_neighborhood,
            labels=labels,
            labels_mask=labels_mask,
            affinities=gt_affs,
            affinities_mask=gt_affs_mask)

    if phase == 'euclid':
        train_pipeline += BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask)

    train_pipeline += DefectAugment(
            raw,
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            artifacts=artifacts,
            artifacts_mask=artifacts_mask,
            contrast_scale=0.5,
            axis=0)

    train_pipeline += IntensityScaleShift(raw, 2,-1)
    train_pipeline += PreCache(
            cache_size=40,
            num_workers=10)

    train_pipeline += Predict(
            checkpoint='../setup02/train_net_checkpoint_400000',
            graph='sd_net.meta',
            inputs={
                sd_config['raw']: raw
            },
            outputs={
                sd_config['embedding']: pretrained_lsd
            })

    train_inputs = {
        context_config['raw']: raw_cropped,
        context_config['pretrained_lsd']: pretrained_lsd,
        context_config['gt_embedding']: gt_embedding,
        context_config['loss_weights_embedding']: gt_embedding_scale,
        context_config['gt_affs']: gt_affs
    }
    if phase == 'euclid':
        train_loss = context_config['loss']
        train_optimizer = context_config['optimizer']
        train_summary = context_config['summary']
        train_inputs[context_config['loss_weights_affs']] = gt_affs_scale
    else:
        train_loss = None
        train_optimizer = add_malis_loss
        train_inputs['gt_seg:0'] = labels
        train_inputs['gt_affs_mask:0'] = gt_affs_mask
        train_summary = 'setup49_malis_loss:0'

    train_pipeline += Train(
            'train_net',
            optimizer=train_optimizer,
            loss=train_loss,
            inputs=train_inputs,
            outputs={
                context_config['embedding']: embedding,
                context_config['affs']: affs
            },
            gradients={},
            summary=train_summary,
            log_dir='log', 
            save_every=10000)

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)
    train_pipeline += Snapshot({
                raw_cropped: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_embedding: 'volumes/gt_embedding',
                embedding: 'volumes/pred_embedding',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities'
            },
            dataset_dtypes={
                labels: np.uint64
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
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
