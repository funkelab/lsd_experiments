import daisy
import neuroglancer
import numpy as np
import sys

from funlib.show.neuroglancer import add_layer, ScalePyramid

f='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(17)
]

f = sys.argv[1]

diffs = daisy.open_ds(f, 'volumes/lsd_diffs')

seg = [
    daisy.open_ds(f, 'volumes/segmentation/s%d'%s)
    for s in range(9)
]

lsds = daisy.open_ds(f, 'volumes/lsds/s0')


viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, diffs, 'diffs')
    add_layer(s, raw, 'raw')
    add_layer(s, lsds, 'mean', shader='rgb')
    add_layer(s, seg, 'seg')
print(viewer)
