import daisy
import neuroglancer
import numpy as np
import sys

from funlib.show.neuroglancer import add_layer, ScalePyramid

class EmptyMeshGenerator():
    @staticmethod
    def get_mesh(object_id):
        raise neuroglancer.local_volume.MeshImplementationNotAvailable()

neuroglancer.set_server_bind_address('0.0.0.0')

f='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(17)
]

f = sys.argv[1]

affs_glia = [
    daisy.open_ds(f, 'volumes/affs/s%d'%s)
    for s in range(9)
]

seg = [
    daisy.open_ds(f, 'volumes/segmentation_threshold_debug/s%d'%s)
    for s in range(9)
]

frags = daisy.open_ds(f, 'volumes/fragments')

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
    add_layer(s, seg, 'seg')
    add_layer(s, frags, 'frags')
    add_layer(s, raw, 'raw')
    add_layer(s, affs_glia, 'affs_glia', shader='rgb')
    # s.navigation.position.voxelCoordinates = (114481, 35485, 4549)
    # s.layers['frags'].source._mesh_generator = EmptyMeshGenerator
    # s.layers['seg'].source._mesh_generator = EmptyMeshGenerator
print(viewer)
