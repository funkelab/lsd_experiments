import daisy
import neuroglancer
import numpy as np
import operator

neuroglancer.set_server_bind_address('0.0.0.0')


f='/groups/funke/funkelab/sheridana/lsd_experiments/cremi/03_predict/setup55_g/300000/testing/sample_C_padded_20160501.aligned.filled.cropped.62:153.n5'

xyz_resolution = [4,4,40]
max_xyz = [248156, 133718, 7062]

raw = [
    daisy.open_ds(f, 'volumes/affs'),
    daisy.open_ds(f, 'volumes/s1'),
    daisy.open_ds(f, 'volumes/s2'),
    daisy.open_ds(f, 'volumes/s3'),
    daisy.open_ds(f, 'volumes/s4'),
    daisy.open_ds(f, 'volumes/s5'),
    daisy.open_ds(f, 'volumes/s6'),
    daisy.open_ds(f, 'volumes/s7'),
    daisy.open_ds(f, 'volumes/s8'),
    daisy.open_ds(f, 'volumes/s9')
]

f='/nrs/funke/sheridana/fafb/setup06/400000/test_lsds_fafb.n5'
lsds = daisy.open_ds(f, 'volumes/lsds')

class ScalePyramid(neuroglancer.LocalVolume):

    def __init__(self, volume_layers):

        super(neuroglancer.LocalVolume, self).__init__()

        for l in volume_layers:
            print("volume layer voxel_size: ", l.voxel_size)

        self.min_voxel_size = min(
            [
                tuple(l.voxel_size)
                for l in volume_layers
            ]
        )
        print("min_voxel_size: ", self.min_voxel_size)

        self.volume_layers = {
            tuple(map(operator.truediv, l.voxel_size, self.min_voxel_size)): l
            for l in volume_layers
        }
        print("scale keys: ", self.volume_layers.keys())

        print(self.info())

    @property
    def volume_type(self):
        return self.volume_layers[(1,1,1)].volume_type

    @property
    def token(self):
        return self.volume_layers[(1,1,1)].token

    def info(self):

        scales = []

        # for (sx, sy, sz) in ... 
        for scale, layer in sorted(self.volume_layers.items()):

            # TODO: support 2D
            scale_info = layer.info()['threeDimensionalScales'][0]
            scale_info['key'] = ','.join('%d'%s for s in scale)
            scales.append(scale_info)

        reference_layer = self.volume_layers[(1, 1, 1)]

        info = {
            'volumeType': reference_layer.volume_type,
            'dataType': reference_layer.data_type,
            'maxVoxelsPerChunkLog2': 20,    # Default is 18
            'encoding': reference_layer.encoding,
            'numChannels': reference_layer.num_channels,
            'generation': reference_layer.change_count,
            'threeDimensionalScales': scales
        }

        return info

    def get_encoded_subvolume(self, data_format, start, end, scale_key='1,1,1'):

        # print("start: ", start)
        # print("end  : ", end)
        # print("scale: ", scale_key)

        scale = tuple(int(s) for s in scale_key.split(','))

        return self.volume_layers[scale].get_encoded_subvolume(
            data_format,
            start,
            end,
            scale_key='1,1,1')

    def get_object_mesh(self, object_id):
        return self.volume_layers[(1,1,1)].get_object_mesh(object_id)

    def invalidate(self):
        return self.volume_layers[(1,1,1)].invalidate()

def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    is_multiscale = type(a) == list

    if is_multiscale:

        for v in a:
            print("voxel size: ", v.voxel_size)

        layer = ScalePyramid(
            # TODO: get scale level of a[i]
            [
                neuroglancer.LocalVolume(
                    data=v.data,
                    offset=v.roi.get_offset()[::-1],
                    voxel_size=v.voxel_size[::-1])
                for v in a
            ])
    else:
        layer = neuroglancer.LocalVolume(
            data=a.data,
            offset=a.roi.get_offset()[::-1],
            voxel_size=a.voxel_size[::-1])

    s.layers.append(
            name=name,
            layer=layer,
            **kwargs)

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add(s, raw, 'raw')
    add(s, lsds, 'lsds', shader='rgb')
print(viewer)
