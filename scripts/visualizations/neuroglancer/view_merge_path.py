import daisy
import h5py
import neuroglancer
import numpy as np
import itertools
import operator
import sys
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

f = sys.argv[1]

# affs = [
    # daisy.open_ds(f, 'volumes/affs/s%d'%s)
    # for s in range(9)
# ]

seg = [
    daisy.open_ds(f, 'volumes/segmentation_validation/s%d'%s)
    for s in range(9)
]

affs = daisy.open_ds(f, 'volumes/affs')
fragments = daisy.open_ds(f, 'volumes/fragments')

debug_file = sys.argv[2]
debug_fragments = daisy.open_ds(debug_file, 'volumes/fragments')

f = sys.argv[3]

centers = np.load(f)['centers']
print(centers)
nodes = []
edge_nodes = []
edges = []

def to_ng_coords(coords):
    return np.flip(coords).astype(np.float32) + 0.5

for (u, v) in centers:
    u_site = to_ng_coords(u)
    v_site = to_ng_coords(v)

    print(u_site, v_site)

    nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=u_site,
                radii=(100,100,100),
                id=next(ngid)
                )
            )
    edges.append(
            neuroglancer.LineAnnotation(
                point_a=u_site,
                point_b=v_site,
                id=next(ngid)
                )
            )
    t = 0
    l = np.linalg.norm(v_site - u_site)
    while t < l:
        p = u_site + (t/l)*(v_site - u_site)
        t += 30
        edge_nodes.append(
                neuroglancer.EllipsoidAnnotation(
                    center=p,
                    radii=(30,30,30),
                    id=next(ngid)
                    )
                )
nodes.append(
        neuroglancer.EllipsoidAnnotation(
            center=to_ng_coords(centers[-1][1]),
            radii=(300,300,300),
            id=next(ngid)
            )
        )

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    print('Adding layers to viewer...')
    add_layer(s, affs, 'affs', shader='rgb')
    add_layer(s, seg, 'neurons')
    add_layer(s, fragments, 'fragments')
    # add_layer(s, debug_fragments, 'debug fragments')
    add_layer(s, raw, 'raw')

    # s.layers['edges'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#add8e6',
            # annotations=edges,
            # )
    s.layers['nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=nodes,
            )
    s.layers['edge_nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#000000',
            annotations=edge_nodes,
            )
    s.navigation.position.voxelCoordinates = (116250, 36594, 4453) 
print(viewer)
