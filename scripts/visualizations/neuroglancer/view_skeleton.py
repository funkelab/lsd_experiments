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

db_host ='mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke' 
db_name = 'synful'

roi = daisy.Roi(
        (158000, 121800, 403560),
        (76000, 52000, 64000))

print('Loading raw file...')
f='/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(17)
]

f = sys.argv[1]

seg = [
    daisy.open_ds(f, 'volumes/segmentation_threshold_debug/s%d'%s)
    for s in range(9)
]

fragments = daisy.open_ds(f, 'volumes/fragments')

f = sys.argv[2]

centers = np.load(f)['centers']
nodes = []
edge_nodes = []
edges = []

gt_nodes = []
gt_edge_nodes = []

def to_ng_coords(coords):
    return np.flip(coords).astype(np.float32) + 0.5

for (u, v) in centers:
    u_site = to_ng_coords(u)
    v_site = to_ng_coords(v)

    nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=u_site,
                radii=(50,50,50),
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
            radii=(50,50,50),
            id=next(ngid)
            )
        )

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

skeleton = get_skeleton_from_db(422272)

for u, v in skeleton.edges():
    u = skeleton.nodes[u]
    v = skeleton.nodes[v]

    if 'x' not in u or 'x' not in v:
       continue

    pos_u = [u['x'], u['y'], u['z']]
    pos_v = [v['x'], v['y'], v['z']]

    gt_nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=pos_u,
                radii=(100,100,100),
                id=next(ngid)
                )
            )
    t = 0
    l = np.linalg.norm(v_site - u_site)
    while t < l:
        p = u_site + (t/l)*(v_site - u_site)
        t += 30
        gt_edge_nodes.append(
                neuroglancer.EllipsoidAnnotation(
                    center=p,
                    radii=(30,30,30),
                    id=next(ngid)
                    )
                )

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    print('Adding layers to viewer...')
    add_layer(s, seg, 'neurons')
    add_layer(s, raw, 'raw')

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
    s.layers['gt_nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=gt_nodes,
            )
    s.layers['gt_edge_nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#32CD32',
            annotations=gt_edge_nodes,
            )

print(viewer)
