import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# logging.basicConfig(filename='debug.log', filemode='w', level=logging.DEBUG)

data_dir = '../../01_data/'

sample = 'test_volume.zarr'

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def get_padding(shape):

    padding = [
            int(round(np.sqrt(np.sum([i*i
                for i in [s,]*3]))/2))
                for s in shape
            ]

    return padding

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
    unlabelled = ArrayKey('UNLABELLED')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    gt_affs_mask = ArrayKey('GT_AFFS_MASK')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    voxel_size = Coordinate((93, 62, 62))

    input_shape = config['input_shape']
    output_shape = config['output_shape']

    input_size = Coordinate(input_shape)*voxel_size
    output_size = Coordinate(output_shape)*voxel_size

    context = Coordinate(get_padding(output_shape))*voxel_size
    logging.info(context)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_scale, output_size)
    request.add(gt_affs_mask, output_size)

    snapshot_request = BatchRequest({
        affs: request[gt_affs],
        affs_gradient: request[gt_affs]
    })

    pos_sources = ZarrSource(
                    os.path.join(data_dir, sample),
                    datasets = {
                        raw: 'train/raw_clahe/s0',
                        labels: 'train/artifact/s0',
                        labels_mask: 'train/labels_mask/s0',
                        unlabelled: 'train/unlabelled_artifact/s0',
                    },
                    array_specs = {
                        raw: ArraySpec(interpolatable=True),
                        labels: ArraySpec(interpolatable=False),
                        labels_mask: ArraySpec(interpolatable=False),
                        unlabelled: ArraySpec(interpolatable=False)
                    }
                )
    pos_sources += Normalize(raw)
    pos_sources += Pad(labels, context)
    pos_sources += Pad(labels_mask, context)
    pos_sources += Pad(unlabelled, context)
    pos_sources += RandomLocation(mask=unlabelled, min_masked=0.3)

    neg_rois = [
            [(14415,11036,6758), (10788,20584,20212)],
            [(9300,75330,6820), (13950,24242,24862)],
            [(11625,35836,82150), (11160,21638,24118)],
            [(0,93620,68386), (11160,23312,27714)],
            [(0,17422,133114), (18600,19778,19034)]
        ]

    neg_sources = tuple(
                ZarrSource(
                    os.path.join(data_dir, sample),
                    datasets = {
                        raw: 'train/raw_clahe/s0',
                        labels: 'train/artifact/s0',
                        labels_mask: 'train/labels_mask/s0',
                        unlabelled: 'train/unlabelled_artifact/s0',
                    },
                    array_specs = {
                        raw: ArraySpec(interpolatable=True),
                        labels: ArraySpec(interpolatable=False),
                        labels_mask: ArraySpec(interpolatable=False),
                        unlabelled: ArraySpec(interpolatable=False)
                    }
                ) +
                Crop(raw, gunpowder.Roi(roi[0], roi[1])) +
                Crop(labels, gunpowder.Roi(roi[0], roi[1])) +
                Crop(labels_mask, gunpowder.Roi(roi[0], roi[1])) +
                Crop(unlabelled, gunpowder.Roi(roi[0], roi[1])) +
                Normalize(raw) +
                Pad(raw, None) +
                Pad(labels, context) +
                Pad(labels_mask, context) +
                Pad(unlabelled, context) + 
                RandomLocation(mask=labels_mask, min_masked=0.5)
                for roi in neg_rois
            )

    neg_sources += RandomProvider()

    train_pipeline = (
        (pos_sources, neg_sources) +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,4,6],
            jitter_sigma=[0,1,1],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=5,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True)+
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
                config['gt_affs']: gt_affs,
                config['loss_weights_affs']: gt_affs_scale
            },
            outputs={
                config['affs']: affs
            },
            gradients={
                config['affs']: affs_gradient
            },
            summary=config['summary'],
            log_dir='log',
            save_every=10000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/artifact_ids',
                labels_mask: 'volumes/labels/labels_mask',
                gt_affs: 'volumes/gt_affs',
                affs: 'volumes/pred_affs',
                affs_gradient: 'volumes/affs_gradient',
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
