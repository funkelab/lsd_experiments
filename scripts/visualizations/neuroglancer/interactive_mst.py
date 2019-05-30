from __future__ import print_function

import neuroglancer
import sys
import daisy
import find_merge_path
import numpy as np
import itertools
from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=1)

raw_file = '/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'

raw = [
    daisy.open_ds(raw_file, 'volumes/raw/s%d'%s)
    for s in range(17)
]

f = sys.argv[1]

frags = daisy.open_ds(f, 'volumes/fragments')
seg = [
    daisy.open_ds(f, 'volumes/segmentation_validation/s%d'%s)
    for s in range(9)
]
affs = daisy.open_ds(f, 'volumes/affs')

frag_list = []
nodes = []
edge_nodes = []

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, seg, 'seg')
    add_layer(s, frags, 'frags')
    add_layer(s, affs, 'affs', shader='rgb')
    add_layer(s, raw, 'raw')
    s.navigation.position.voxelCoordinates = (115511, 36341, 4476)

def get_graph(s):
    print('  Mouse position: %s' % (s.mouse_voxel_coordinates,))
    print('  Layer selected values: %s' % (s.selected_values,))

    frag_1 = int(''.join(map(str, frag_list)))
    frag_2 = s.selected_values['frags']

    find_merge_path.find_path(find_merge_path.read_graph(s.selected_values['seg']),frag_1, frag_2)

    npz = '%s_%s.npz' % (frag_1, frag_2) 

    centers = np.load(npz)['centers']

    for (u, v) in centers:
        u_site = to_ng_coords(u)
        v_site = to_ng_coords(v)

        nodes.append(
                neuroglancer.EllipsoidAnnotation(
                    center=u_site,
                    radii=(100,100,100),
                    id=next(ngid)
                    )
                )
        t = 0
        l = np.linalg.norm(v_site - u_site)
        while t < l:
            p = u_site + (t/l)*(v_site - u_site)
            t+=30
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

def add_mst(s):
    with viewer.txn() as s:
        s.layers['nodes'] = neuroglancer.AnnotationLayer(
                voxel_size=(1,1,1),
                filter_by_segmentation=False,
                annotation_color='#ff00ff',
                annotations=nodes,
                )
        s.layers['edge_nodes'] = neuroglancer.AnnotationLayer(
                voxel_size=(1,1,1),
                filter_by_segmentation=False,
                annotation_color='#32CD32',
                annotations=edge_nodes,
                )

def get_first_frag(s):

    print(' Frag 1 is: %s' % (s.selected_values['frags']))
    first_frag = s.selected_values['frags']
    frag_list.append(first_frag)

def to_ng_coords(coords):
    return np.flip(coords).astype(np.float32) + 0.5

viewer.actions.add('get-first-frag', get_first_frag)
viewer.actions.add('get-graph', get_graph)
viewer.actions.add('add-mst', add_mst)
with viewer.config_state.txn() as s:
    s.input_event_bindings.viewer['keyt'] = 'get-first-frag'
    s.input_event_bindings.viewer['keyy'] = 'get-graph'
    s.input_event_bindings.viewer['keyu'] = 'add-mst'

print(viewer)


