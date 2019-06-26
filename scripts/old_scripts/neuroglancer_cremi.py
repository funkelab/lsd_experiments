import daisy
import neuroglancer
import numpy as np

neuroglancer.set_server_bind_address('0.0.0.0')

f = '/groups/funke/funkelab/sheridana/lsd_experiments/cremi/03_predict/setup41/300000/testing/sample_C_padded_20160501.aligned.filled.cropped.62:153.n5'
affs = daisy.open_ds(f, 'volumes/affs')

f = '/groups/funke/funkelab/sheridana/lsd_experiments/cremi/03_predict/setup06/400000/testing/sample_C_padded_20160501.aligned.filled.cropped.62:153.n5'
affs = daisy.open_ds(f, 'volumes/lsds')

f = '/groups/funke/funkelab/sheridana/lsd_experiments/cremi/01_data/sample_C+_padded_20160908.aligned.n5'
raw = daisy.open_ds(f, 'volumes/raw')

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

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add(s, lsds, 'lsds', shader='rgb')
    add(s, affs, 'affs', shader='rgb')
    add(s, raw, 'raw')
print(viewer)
