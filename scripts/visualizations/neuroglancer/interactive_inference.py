from __future__ import print_function

import neuroglancer
import daisy
import json
import numpy as np
import os
import sys
import time

sys.path.append('/groups/funke/funkelab/sheridana/src/scale_pyramid')
from scale_pyramid import create_scale_pyramid

sys.path.append('/groups/funke/funkelab/sheridana/lsd_experiments/scripts')
from test_blockwise_prediction import predict_blockwise
from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

experiment = 'cremi'
setup = 'setup61_p'
iteration = 400000
raw_file = 'inference_roi.json'
raw_dataset = 'volumes/raw/'
out_file = 'interactive_inference'
file_name = 'test.zarr'
num_workers = 10
db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'interactive_inference'

config_file = 'predict.json'
voxel_size = [4,4,40]
volume_size = [10000,10000,10000]

create_pyramid = True
scales = [(1,2,2), (1,2,2), (1,2,2), (2,2,2), (2,2,2), (2,2,2), (2,2,2), (2,2,2)]
chunk_shape = [128, 128, 128]

add_affs = False
add_lsds = False


# fafb_raw='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'
f = sys.argv[1]
raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(7)
]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    add_layer(s, raw, 'raw')

def run_inference(s):
    coords = [int(i) for i in to_ng_coords(s.mouse_voxel_coordinates)]
    print(coords)

    roi_config = {
            'container': fafb_raw,
            'offset': coords,
            'size': volume_size
    }

    with open(raw_file, 'w') as f:
        json.dump(roi_config, f)

    predict_config = {
            'experiment': experiment,
            'setup': setup,
            'iteration': iteration,
            'raw_file': raw_file,
            'raw_dataset': raw_dataset,
            'out_file': out_file,
            'file_name': file_name,
            'num_workers': num_workers,
            'db_host': db_host,
            'db_name': db_name
    }

    with open(config_file, 'w') as f:
        json.dump(predict_config, f)

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    predict_blockwise(**config)
    print('Total time to predict: %f seconds' %(time.time() - start))

    f = os.path.join(out_file, setup, str(iteration), file_name)
    affs = 'volumes/affs'
    lsds = 'volumes/lsds'

    if create_pyramid:
        print('Creating scale pyramid...')
        create_scale_pyramid(f, affs, scales, chunk_shape)

        affs_layer = [
                daisy.open_ds(f, 'volumes/affs/s%d'%s)
                for s in range(9)
        ]

    else:
        affs_layer = daisy.open_ds(f, affs)

    with viewer.txn() as s:
        if add_affs:
            print('Adding layers to session...')
            add_layer(s, affs_layer, 'affs', shader='rgb')

def to_ng_coords(coords):
    coords = [round(i)*j for i,j in zip(coords, voxel_size)]
    return np.flip(coords).astype(np.float32)

viewer.actions.add('run_inference', run_inference)

with viewer.config_state.txn() as s:
    s.input_event_bindings.viewer['keyt'] = 'run_inference'

print(viewer)

