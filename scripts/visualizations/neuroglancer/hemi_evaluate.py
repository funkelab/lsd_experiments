import daisy
import neuroglancer
import numpy as np
import sys
import zarr

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

f = sys.argv[1]

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
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

f = sys.argv[2]

mtlsd_affs = [
    daisy.open_ds(f, 'volumes/affs/s%d'%s)
    for s in range(9)
]

mtlsd_frags = daisy.open_ds(f, 'volumes/fragments')

mtlsd_seg = [
    daisy.open_ds(f, 'volumes/segmentation_64/s%d'%s)
    for s in range(9)
]

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, raw, 'raw')
    add_layer(s, gt, 'original gt')
    add_layer(s, relabelled, 'relabelled')
    add_layer(s, mtlsd_affs, 'mtlsd affs', shader='rgb')
    add_layer(s, mtlsd_frags, 'mtlsd frags')
    add_layer(s, mtlsd_seg, 'mtlsd seg')
print(viewer)
