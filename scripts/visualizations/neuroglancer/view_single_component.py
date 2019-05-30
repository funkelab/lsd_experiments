import daisy
import h5py
import neuroglancer
import numpy as np
import itertools
import operator
import sys
from funlib.show.neuroglancer import ScalePyramid, add_layer

db_host ='mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke' 
db_name = 'synful'

roi = daisy.Roi(
        (158000, 121800, 403560),
        (76000, 52000, 64000))


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

affs = [
    daisy.open_ds(f, 'volumes/affs/s%d'%s)
    for s in range(9)
]

seg = [
    daisy.open_ds(f, 'volumes/segmentation_threshold_debug/s%d'%s)
    for s in range(9)
]

fragments = daisy.open_ds(f, 'volumes/fragments')

def get_skeleton_from_db(component_id):

    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection='calyx_catmaid.nodes',
            edges_collection='calyx_catmaid.edges',
            mode='r',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'],
            node_attribute_collections={
                'calyx_neuropil_mask': ['masked'],
                'calyx_neuropil_components': ['component_id']
            })

    skeleton = skeletons_provider.get_graph(
            roi=roi,
            nodes_filter={'component_id': component_id}
            )

    return skeleton

nodes = []
edges = []

for comp in (91, 422272):
    skeleton = get_skeleton_from_db(comp)

    for u, v in skeleton.edges():
        u = skeleton.nodes[u]
        v = skeleton.nodes[v]

        if 'x' not in u or 'x' not in v:
           continue

        pos_u = [u['x'], u['y'], u['z']]
        pos_v = [v['x'], v['y'], v['z']]

        nodes.append(
                neuroglancer.EllipsoidAnnotation(
                    center=pos_u,
                    radii=(30,30,30),
                    id=next(ngid)
                    )
                )
        edges.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u,
                    point_b=pos_v,
                    id=next(ngid)
                    )
                )

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    print('Adding layers to viewer...')
    add_layer(s, affs, 'affs', shader='rgb')
    add_layer(s, seg, 'neurons')
    add_layer(s, fragments, 'fragments')
    add_layer(s, raw, 'raw')

    s.layers['edges'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#add8e6',
            annotations=edges,
            )
    s.layers['nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=nodes,
            )
    s.navigation.position.voxelCoordinates = (112514, 40396, 4926)
print(viewer)
