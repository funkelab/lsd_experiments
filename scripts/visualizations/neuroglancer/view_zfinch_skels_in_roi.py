import daisy
import itertools
import json
import neuroglancer
import numpy as np
import operator
import os
import sys

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=1)
db_host = "mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke"
db_name = "zebrafinch_gt_skeletons_new_gt_9_9_20_testing"

f = sys.argv[1]

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(6)
]

# f = sys.argv[2]

# seg = [
    # daisy.open_ds(f, 'volumes/segmentation_6_62/s%d'%s)
    # for s in range(6)
# ]


def to_ng_coords(coords):
    return np.flip(coords).astype(np.float32) + 0.5

def get_skeletons_from_db(roi):

    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection='zebrafinch.nodes',
            edges_collection='zebrafinch.edges',
            mode='r',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'])

    skeletons = skeletons_provider.get_graph(
            roi=roi
            )

    return skeletons

nodes = []
edges = []
gt_edge_nodes = []

voxel_size = [20,9,9]

path_to_configs = sys.argv[2]

with open(os.path.join(path_to_configs, 'zebrafinch_11_micron_roi_6.json'), 'r') as f:
    config = json.load(f)



roi_offset = config['roi_offset']
roi_shape = config['roi_shape']

roi = daisy.Roi(roi_offset, roi_shape)

skeletons = get_skeletons_from_db(roi)

for u, v in skeletons.edges():
    u = skeletons.nodes[u]
    v = skeletons.nodes[v]

    if 'x' not in u or 'x' not in v:
       continue

    pos_u = [u['z'], u['y'], u['x']]
    pos_v = [v['z'], v['y'], v['x']]

    pos_u = to_ng_coords(pos_u)
    pos_v = to_ng_coords(pos_v)

    nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=pos_u,
                radii=(100,100,100),
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

    t = 0
    l = np.linalg.norm(pos_v - pos_u)
    while t < l:
        p = pos_u + (t/l)*(pos_v - pos_u)
        t+=30
        gt_edge_nodes.append(
                neuroglancer.EllipsoidAnnotation(
                    center=p,
                    radii=(50,50,50),
                    id=next(ngid)
                    )
                )

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    # s.layers['edges'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#ff00ff',
            # annotations=edges,
            # )
    s.layers['nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#5280e9',
            annotations=nodes,
            )
    s.layers['gt_edge_nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=gt_edge_nodes,
            )
    # add_layer(s, seg, 'seg')
    add_layer(s, raw, 'raw')
print(viewer)
