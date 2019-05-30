import daisy
import h5py
import neuroglancer
import numpy as np
import itertools
import operator
from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=1)

print('Loading raw file...')
f='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'

xyz_resolution = [4,4,40]
max_xyz = [248156, 133718, 7062]

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(17)
]

f = '/nrs/funke/sheridana/fafb/setup55/400000/block2.n5'
neurons = daisy.open_ds(f, '/volumes/segmentation_full')

print('Loading synapses...')
f=h5py.File('/nrs/funke/buhmannj/calyx/evaluation/db_syntist_001/all_KC_pred_syns.hdf', 'r')

locations = f['annotations/locations']

id_mapping = dict(zip(f['annotations/ids'], locations))

(pre_sites, post_sites, connectors) = ([], [], [])

print('Mapping synapse ids to locations...')

for (pre, post) in f['annotations/presynaptic_site/partners'][0:10000]:
    pre_site = np.flip(id_mapping[pre])
    post_site = np.flip(id_mapping[post])

    print('Appending pre location: ', pre_site)
    print('Appending post location: ', post_site)

    pre_sites.append(
            neuroglancer.EllipsoidAnnotation(
                center=pre_site,
                radii=(30,30,30),
                id=next(ngid)
                )
            )
    post_sites.append(
            neuroglancer.EllipsoidAnnotation(
                center=post_site,
                radii=(30,30,30),
                id=next(ngid)
                )
            )
    connectors.append(
            neuroglancer.LineAnnotation(
                point_a=pre_site,
                point_b=post_site,
                id=next(ngid)
                )
            )

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    print('Adding layers to viewer...')
    add_layer(s, neurons, 'neurons')
    add_layer(s, raw, 'raw')

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
    # s.navigation.position.voxelCoordinates = (109249, 38704, 5092)
print(viewer)
