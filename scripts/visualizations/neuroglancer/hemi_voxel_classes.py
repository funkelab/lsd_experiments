import daisy
import neuroglancer
import numpy as np
import sys
import zarr

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

f = sys.argv[1]

raw_large = [
    daisy.open_ds(f, 'volumes/raw_large/s%d'%s)
    for s in range(9)
]

gt = [
    daisy.open_ds(f, 'volumes/labels/neuron_ids/s%d'%s)
    for s in range(9)
]

relabelled = [
    daisy.open_ds(f, 'volumes/labels/relabelled_eroded_ids/s%d'%s)
    for s in range(9)
]

voxel_classes = [
    daisy.open_ds(f, 'volumes/labels/voxel_classes/s%d'%s)
    for s in range(9)
]

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, gt, 'original gt')
    add_layer(s, relabelled, 'relabelled')
    add_layer(s, voxel_classes, 'voxel classes')
    add_layer(s, raw_large, 'raw large')
print(viewer)
