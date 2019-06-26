import z5py
import numpy as np
import neuroglancer

neuroglancer.set_server_bind_address('0.0.0.0')

raw = z5py.File('/nrs/turaga/funkej/fib19/fib19.n5')['volumes/raw/s0']
segmentation = z5py.File('/nrs/turaga/funkej/fib19/fib19_whitelisted.n5')['volumes/labels/whitelisted']

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    s.voxel_size = [10, 10, 10]
    s.layers.append(
            name='raw',
            layer=neuroglancer.LocalVolume(
                data=raw,
                voxel_size=raw.attrs['resolution'][::-1]
                ),
            shader="""
void main() {
    emitGrayscale(toNormalized(getDataValue()));
}
""")
    s.layers.append(name='segmentation',
            layer=neuroglancer.LocalVolume(
                data=segmentation,
                offset=segmentation.attrs['offset'][::-1],
                voxel_size=segmentation.attrs['resolution'][::-1]
                ))
print(viewer)
