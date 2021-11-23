import daisy
import h5py
import neuroglancer
import numpy as np
import itertools
import operator
import sys
from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=0)

db_host ='mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org:27023/admin?replicaSet=rsFunke' 
db_name = 'zebrafinch_gt_skeletons_new_gt_9_9_20_testing'

f = sys.argv[1]

raw = [
    daisy.open_ds(f, 'volumes/raw/s%d'%s)
    for s in range(6)
]

f = sys.argv[2]

frags = daisy.open_ds(f, 'volumes/fragments_32_micron_roi_test/s4')
roi = frags.roi

# seg = [
    # daisy.open_ds(f, 'volumes/segmentation_16/s%d'%s)
    # for s in range(6)
# ]

a_seg_75 = daisy.open_ds(f, 'volumes/segmentation_16/s4')
a_seg_50 = daisy.open_ds(f, 'volumes/segmentation_64/s4')

f = sys.argv[3]

v_seg_75 = daisy.open_ds(f, 'volumes/segmentation_28/s4')
v_seg_50 = daisy.open_ds(f, 'volumes/segmentation_62/s4')

def get_frags(fragments,sites,roi):

    fragments = fragments[roi]
    fragments.materialize()

    fragment_ids = np.array([
        fragments[daisy.Coordinate((site[0],site[1],site[2]))]
        for site in sites
    ])

    fg_mask = fragment_ids != 0
    fragment_ids = fragment_ids[fg_mask]

    for i in fragment_ids:
        print(i)

edges = []

gt_nodes = []
gt_edge_nodes = []
pos_u_nodes = []
pos_v_nodes = []

def to_ng_coords(coords):
    return np.flip(coords).astype(np.float32) + 0.5

def get_skeleton_from_db(component_id):

    print('Opening graph...')

    node_mask = 'zebrafinch_mask_32_micron_roi_not_masked'
    node_components = 'zebrafinch_components_32_micron_roi_not_masked'

    skeletons_provider = daisy.persistence.MongoDbGraphProvider(
            host=db_host,
            db_name=db_name,
            nodes_collection='zebrafinch.nodes',
            edges_collection='zebrafinch.edges',
            mode='r',
            endpoint_names=['source', 'target'],
            position_attribute=['z', 'y', 'x'],
            node_attribute_collections={
                node_mask: ['masked'],
                node_components: ['component_id']})

    skeleton = skeletons_provider.get_graph(
            roi=roi,
            nodes_filter={'component_id': component_id}
            )

    return skeleton

skeleton = get_skeleton_from_db(16329)

sites = []

for u, v in skeleton.edges():
    u = skeleton.nodes[u]
    v = skeleton.nodes[v]

    if 'z' not in u or 'z' not in v:
       continue

    pos_u = [u['z'], u['y'], u['x']]
    pos_v = [v['z'], v['y'], v['x']]

    sites.append(pos_u)

    pos_u = to_ng_coords(pos_u)
    pos_v = to_ng_coords(pos_v)

    pos_u_nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=pos_u,
                radii=(50,50,50),
                id=next(ngid)
                )
            )

    pos_v_nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=pos_v,
                radii=(50,50,50),
                id=next(ngid)
                )
            )

    t = 0
    l = np.linalg.norm(pos_v - pos_u)
    while t < l:
        p = pos_u + (t/l)*(pos_v - pos_u)
        t += 30
        gt_edge_nodes.append(
                neuroglancer.EllipsoidAnnotation(
                    center=p,
                    radii=(30,30,30),
                    id=next(ngid)
                    )
                )

frag_ids = get_frags(frags, sites, roi)

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    print('Adding layers to viewer...')
    add_layer(s, raw, 'raw')
    add_layer(s, frags, 'frags')
    add_layer(s, a_seg_75, 'a seg 75')
    add_layer(s, a_seg_50, 'a seg 50')
    add_layer(s, v_seg_75, 'v seg 75')
    add_layer(s, v_seg_50, 'v seg 50')

  #   s.layers['nodes'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#ff00ff',
            # annotations=nodes,
            # )
  #   s.layers['edge_nodes'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#32CD32',
            # annotations=edge_nodes,
            # )
  #   s.layers['gt_nodes'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#ff00ff',
            # annotations=gt_nodes,
            # )
    # s.layers['gt_edge_nodes'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#32CD32',
            # annotations=gt_edge_nodes,
            # )
    # s.layers['edges'] = neuroglancer.AnnotationLayer(
        # voxel_size=(1,1,1),
        # filter_by_segmentation=False,
        # annotation_color='#32CD32',
        # annotations=edges,
        # )
    s.layers['pos_u_nodes'] = neuroglancer.AnnotationLayer(
            voxel_size=(1,1,1),
            filter_by_segmentation=False,
            annotation_color='#ff00ff',
            annotations=pos_u_nodes,
            )
  #   s.layers['pos_v_nodes'] = neuroglancer.AnnotationLayer(
            # voxel_size=(1,1,1),
            # filter_by_segmentation=False,
            # annotation_color='#32CD32',
            # annotations=pos_v_nodes,
            # )
    # location = [48645, 47097, 52280]
    # location = [48648, 47128, 52260]
    location = sites[0]
    vs = [9, 9, 20]
    s.navigation.position.voxelCoordinates = [i/j for i,j in zip(location, vs)]
    # s.navigation.zoomFactor = 0.5

print(viewer)
