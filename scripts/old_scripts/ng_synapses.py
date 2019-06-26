import h5py
import neuroglancer
import numpy as np
import itertools
import operator

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=1)

def add(s, a, name, shader=None):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=a,
                offset=[0,0,0],
                voxel_size=[4,4,40]
            ),
            **kwargs)


f = h5py.File('/groups/funke/cremi/01_data/sample_C_20160501.hdf', 'r')

raw = f['volumes/raw']
neurons = f['volumes/labels/neuron_ids']
clefts = f['volumes/labels/clefts']

locations = f['annotations/locations']

id_mapping = dict(zip(f['annotations/ids'], locations))

(pre_sites, post_sites, connectors) = ([], [], [])

for (pre, post) in f['annotations/presynaptic_site/partners']:
    pre_site = np.flip(id_mapping[pre])
    post_site = np.flip(id_mapping[post])

    segments = (neurons[pre_site[2]/40, pre_site[1]/4,pre_site[0]/4],neurons[post_site[2]/40,post_site[1]/4, post_site[0]/4])

    pre_sites.append(
            neuroglancer.EllipsoidAnnotation(
                center=pre_site,
                radii=(30,30,30),
                id=next(ngid),
                segments=segments
                )
            )
    post_sites.append(
            neuroglancer.EllipsoidAnnotation(
                center=post_site,
                radii=(30,30,30),
                id=next(ngid),
                segments=segments
                )
            )
    connectors.append(
            neuroglancer.LineAnnotation(
                point_a=pre_site,
                point_b=post_site,
                id=next(ngid),
                segments=segments
                )
            )

viewer = neuroglancer.Viewer()

with viewer.txn() as s: 
    add(s, raw, 'raw')
    add(s, neurons, 'neurons')
    add(s, clefts, 'clefts')

    s.layers['connectors'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#add8e6',
            annotations=connectors,
            )
    s.layers['pre_sites'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#00ff00',
            annotations=pre_sites,
            )
    s.layers['post_sites'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=post_sites,
            )
print(viewer)

