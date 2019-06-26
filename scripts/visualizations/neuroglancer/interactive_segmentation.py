from __future__ import print_function

import neuroglancer
import daisy
import json
import numpy as np
import os
import sys
import time

sys.path.append('/groups/funke/funkelab/sheridana/lsd_experiments/scripts')
import task_helper
from task_04_extract_segmentation import SegmentationTask

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

### Global Input ###

experiment = 'cremi'
db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'interactive_segmentation'
raw_file = 'segment_roi.json'
raw_dataset = 'volumes/raw/s0'
out_file = '/nrs/funke/sheridana/interactive_segmentation/test.zarr'

## Network input ###

setup = 'setup61_p'
iteration = 400000

## Inference ##

num_predict_workers = 5
predict_queue = 'slowpoke'

## Watershed / Agglomerate ##

affs_dataset = "volumes/affs"
fragments_dataset = "volumes/fragments"
block_size = [2000, 2000, 2000]
context = [248, 248, 248]
num_watershed_workers = 5
num_agglom_workers = 5
watershed_queue = 'normal'
agglom_queue = 'normal'
fragments_in_xy = True
epsilon_agglomerate = 0.1
merge_function = 'hist_quant_50'

## Segmentation ##

out_dataset = "volumes/segmentation"
num_segment_workers = 5
threshold = 0.3
segment_queue = 'normal'
edges_collection = 'edges_hist_quant_50'

config_file = 'task_defaults.json'
voxel_size = [4,4,40]
volume_dims = [10000,10000,10000]

fafb = sys.argv[1]
raw = [
    daisy.open_ds(fafb, 'volumes/raw/s%d'%s)
    for s in range(17)
]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    add_layer(s, raw, 'raw')

def segment(s):
    coords = [int(i) for i in to_ng_coords(s.mouse_voxel_coordinates)]
    print(coords)

    roi_config = {
            'container': fafb,
            'offset': coords,
            'size': volume_dims
    }

    with open(raw_file, 'w') as f:
        json.dump(roi_config, f)

    task_config = {

            "GlobalInput":
                {
                    "experiment": experiment,
                    "db_host": db_host,
                    "db_name": db_name,
                    "out_file": out_file,
                    "raw_file": raw_file,
                    "raw_dataset": raw_dataset
                },

            "Network":
                {
                    "setup": setup,
                    "iteration": iteration
                },

            "PredictTask":
                {
                    "num_workers": num_predict_workers,
                    "queue": predict_queue
                },

            "ExtractFragmentsTask":
                {
                    "affs_dataset": affs_dataset,
                    "fragments_dataset": fragments_dataset,
                    "block_size": block_size,
                    "context": context,
                    "num_workers": num_watershed_workers,
                    "queue": watershed_queue,
                    "fragments_in_xy": fragments_in_xy,
                    "epsilon_agglomerate": epsilon_agglomerate
                },

            "AgglomerateTask":
                {
                    "affs_dataset": affs_dataset,
                    "fragments_dataset": fragments_dataset,
                    "block_size": block_size,
                    "context": context,
                    "num_workers": num_agglom_workers,
                    "queue": agglom_queue,
                    "merge_function": merge_function
                },

            "SegmentationTask":
                {
                    "fragments_dataset": fragments_dataset,
                    "out_dataset": out_dataset,
                    "roi_offset": coords,
                    "roi_shape": volume_dims,
                    "num_workers": num_segment_workers,
                    "threshold": threshold,
                    "queue": segment_queue,
                    "edges_collection": edges_collection
                }
    }

    with open(config_file, 'w') as f:
        json.dump(task_config, f)

    start = time.time()

    user_configs, global_config = task_helper.parseConfigs(['task_defaults.json'])

    daisy.distribute(
            [
                {'task': SegmentationTask(global_config=global_config,
                    **user_configs), 'request': None}
            ],
            global_config=global_config)

    volume_size = np.prod(volume_dims)
    print('Total time to segment %s cubic micron volume: %f seconds' %(volume_size, (time.time() - start)))

def to_ng_coords(coords):
    coords = [round(i)*j for i,j in zip(coords, voxel_size)]
    return np.flip(coords).astype(np.float32)

viewer.actions.add('segment', segment)

with viewer.config_state.txn() as s:
    s.input_event_bindings.viewer['keyt'] = 'segment'

print(viewer)

