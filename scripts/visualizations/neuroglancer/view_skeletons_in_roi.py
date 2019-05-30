import daisy
import neuroglancer
import numpy as np
import operator
import sys
import itertools

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=1)
db_host = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke'
db_name = 'synful'

f='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'


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

def get_skeletons_from_db(roi):

    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection='calyx_catmaid.nodes',
            edges_collection='calyx_catmaid.edges',
            mode='r',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'])

    skeletons = skeletons_provider.get_graph(
            roi=roi
            )

    return skeletons

nodes = []
edges = []

roi = daisy.Roi(
        (181320, 140832, 457000),
        (5000, 5000, 5000))

skeletons = get_skeletons_from_db(roi)

for u, v in skeletons.edges():
    u = skeletons.nodes[u]
    v = skeletons.nodes[v]

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
    add_layer(s, seg, 'segmentation')
    add_layer(s, affs, 'affs', shader='rgb')
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
   # # add_layer(s, mask, 'mask')
    # s.navigation.position.voxelCoordinates = (114481, 35485, 4549)
print(viewer)
