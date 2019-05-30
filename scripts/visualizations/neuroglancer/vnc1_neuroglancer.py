import argparse
import daisy
import neuroglancer
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--static-content-url')

args = ap.parse_args()

if args.static_content_url:
    neuroglancer.set_static_content_source(url=args.static_content_url)

neuroglancer.set_server_bind_address('0.0.0.0')

#path to raw
f = "/nrs/funke/funkej/vnc1_t1_1099_4399_htem.zarr"

#raw key
raw = daisy.open_ds(f, 'raw')

#path to labels
f = "/groups/funke/funkelab/sheridana/lsd_experiments/lee/03_process/setup04/540000/16_micron_cutout.zarr"
seg = daisy.open_ds(f, 'volumes/segmentation_full')

f = "/nrs/funke/sheridana/context_test/setup04/540000/extra_training_context_test.zarr"

#affs / segmentation keys
affs = daisy.open_ds(f, 'volumes/affs')
frags = daisy.open_ds(f, 'volumes/fragments')
# seg = daisy.open_ds(f, 'volumes/segmentation')

f = "/nrs/funke/sheridana/context_test/setup04/400000/no_extra_training_context_test.zarr"
affs2 = daisy.open_ds(f, 'volumes/affs/s0')
seg2 = daisy.open_ds(f, 'volumes/segmentation')

f = "/nrs/funke/sheridana/context_test/setup77/360000/auto_context_test.zarr"
affs3 = daisy.open_ds(f, 'volumes/affs')
seg3 = daisy.open_ds(f, 'volumes/segmentation')

f = "/nrs/funke/sheridana/context_test/setup85/360000/glia_mask_context_test.zarr"
affs4 = daisy.open_ds(f, 'volumes/affs')
seg4 = daisy.open_ds(f, 'volumes/segmentation')

f = "/nrs/funke/sheridana/context_test/setup60/400000/long_range_context_test.zarr"
affs5 = daisy.open_ds(f, 'volumes/affs')
seg5 = daisy.open_ds(f, 'volumes/segmentation')

f = "/nrs/funke/sheridana/context_test/setup90/400000/lsd_context_test.zarr"
lsds = daisy.open_ds(f, 'volumes/lsds')

def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a.data,
                offset=a.roi.get_offset()[::-1],
                voxel_size=a.voxel_size[::-1]
            ),
            **kwargs)

viewer = neuroglancer.UnsynchronizedViewer()
with viewer.txn() as s:
    add(s, affs2, 'No Extra training affs', shader='rgb')
    add(s, affs, 'Extra training affs', shader='rgb')
    add(s, affs3, 'MTLSD auto affs', shader='rgb')
    add(s, affs4, 'Glia mask auto affs', shader='rgb')
    # add(s, frags, 'fragments')
    add(s, seg2, 'No extra training seg')
    add(s, seg, 'original seg')
    add(s, seg3, 'MTLSD auto seg')
    add(s, seg4, 'Glia mask auto seg')
    add(s, lsds, 'lsds', shader='rgb')
    add(s, raw, 'raw')
print(viewer)
