import argparse
import daisy
import neuroglancer
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--static-content-url')

args = ap.parse_args()

neuroglancer.set_server_bind_address('0.0.0.0')

if args.static_content_url:
    neuroglancer.set_static_content_source(url=args.static_content_url)

#path to raw
f = "/nrs/funke/funkej/cb2_2.zarr"

#raw key
raw = daisy.open_ds(f, 'raw')

f = '/groups/funke/funkelab/nguyent3/lsd_cb2_old_daisy/cutout/03_process/setup04/118000/cutout.zarr'
seg1 = daisy.open_ds(f, 'volumes/segmentation_0.4')

f = "/nrs/funke/sheridana/context_test/setup90/400000/lsd_context_test_cb2.zarr"
lsds = daisy.open_ds(f, 'volumes/lsds')

f = "/nrs/funke/sheridana/context_test/setup77/390000/auto_context_test_cb2.zarr"
affs = daisy.open_ds(f, 'volumes/affs')
seg2 = daisy.open_ds(f, 'volumes/segmentation')

f = "/nrs/funke/sheridana/context_test/setup85/400000/glia_mask_context_test_cb2.zarr"
affs2 = daisy.open_ds(f, 'volumes/affs')
seg3 = daisy.open_ds(f, 'volumes/segmentation')

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
    add(s, lsds, 'lsds', shader='rgb')
    add(s, affs, 'autocontext affs', shader='rgb')
    add(s, affs2, 'glia mask affs', shader='rgb')
    add(s, seg1, 'Original seg')
    add(s, seg2, 'Autocontext seg')
    add(s, seg3, 'Glia mask seg')
    add(s, raw, 'raw')
    # s.navigation.position.voxelCoordinates = (109249, 38704, 5092)
print(viewer)
