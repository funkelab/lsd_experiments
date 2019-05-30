import daisy
import neuroglancer
import numpy as np
import operator
import sys
import itertools

from synful import database

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

ngid = itertools.count(start=1)

def add_db_synapses(s, db_name, db_host, db_col_name, source_roi, nodes=None):
    dag_db = database.DAGDatabase(db_name, db_host, db_col_name=db_col_name,
                                  mode='r')
    if nodes is not None:
        connector_nodes = []
        source_nodes = []
        for node in nodes:
            if node['type'] == 'connector':
                connector_nodes.append(node)
                source_nodes.append(node['id'])
        edges = dag_db.read_edges(source_ids=source_nodes)
    else:
        edges = dag_db.read_edges(source_roi)
        nodes = dag_db.read_nodes_based_on_edges(edges)
    pos_dic = {}
    for node in nodes:
        node_id = node['id']
        pos = np.array(node['position'])
        pos_dic[node_id] = np.flip(pos)

    pre_sites = []
    post_sites = []
    connectors = []

    for edge in edges:
        u = edge['source']
        v = edge['target']
        if u in pos_dic and v in pos_dic:
            pre_site = pos_dic[u]
            post_site = pos_dic[v]

            pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
                                                              radii=(
                                                                  30, 30, 30),
                                                              id=next(ngid)))
            post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                               radii=(
                                                                   30, 30, 30),
                                                               id=next(ngid)))
            connectors.append(
                neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                            id=next(ngid)))

    s.layers['connectors'] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color='#ffff00',
        annotations=connectors,
    )
    s.layers['pre_sites'] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color='#00ff00',
        annotations=pre_sites,
    )
    s.layers['post_sites'] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color='#ff00ff',
        annotations=post_sites,
    )


def add_db_neuron(s, db_name, db_host, db_col_name, source_roi):
    dag_db = database.DAGDatabase(db_name, db_host, db_col_name=db_col_name,
                                  mode='r')
    edges = dag_db.read_edges(source_roi)
    nodes = dag_db.read_nodes_based_on_edges(edges)

    pos_dic = {}

    for node in nodes:
        node_id = node['id']
        pos = np.array(node['position'])
        pos_dic[node_id] = np.flip(pos)

    pre_sites = []
    post_sites = []
    connectors = []

    for edge in edges:
        u = edge['source']
        v = edge['target']
        if u in pos_dic and v in pos_dic:
            pre_site = pos_dic[u]
            post_site = pos_dic[v]

            pre_sites.append(neuroglancer.EllipsoidAnnotation(center=pre_site,
                                                              radii=(
                                                                  30, 30, 30),
                                                              id=next(ngid)))
            post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                               radii=(
                                                                   30, 30, 30),
                                                               id=next(ngid)))
            connectors.append(
                neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                            id=next(ngid)))

    s.layers['neuron_con'] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color='#8B0000',
        annotations=connectors,
    )
    s.layers['node_u'] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color='#8B0000',
        annotations=pre_sites,
    )
    s.layers['node_v'] = neuroglancer.AnnotationLayer(
        voxel_size=(1, 1, 1),
        filter_by_segmentation=False,
        annotation_color='#8B0000',
        annotations=post_sites,
    )
    return nodes

raw_file = '/groups/futusa/futusa/projects/fafb/v14_align_tps_20170818_dmg.n5'
raw = [
    daisy.open_ds(raw_file, 'volumes/raw/s%d' % s)
    for s in range(17)
]

vanilla_seg_file = '/nrs/funke/sheridana/paper_runs/vanilla_euclidean/' \
                   '3_sample/all_labels/no_autocontext/setup58_p/400000/calyx.zarr'

vanilla_seg = [
    daisy.open_ds(vanilla_seg_file,
                  'volumes/segmentation_threshold_debug/s%d' % s)
    for s in range(9)
]

mtlsd_seg_file = '/nrs/funke/sheridana/paper_runs/mtlsd/6_sample/no_glia/' \
                 'no_autocontext/setup101_p/400000/calyx.zarr'

mtlsd_seg = [
    daisy.open_ds(mtlsd_seg_file,
                  'volumes/segmentation_threshold_debug/s%d' % s)
    for s in range(9)
]

db_name = 'synful'
db_host = 'c04u01'
db_col_name = 'calyx_catmaid'

selected_roi = daisy.Roi(offset=(192000, 132000, 420000),
                         shape=(8000, 8000, 8000))

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    # Synapse database has only edges, and no nodes. So Nodes from skeletons
    # are reused when visualizing synapses.
    nodes = add_db_neuron(s, db_name, db_host, db_col_name,
                          selected_roi)
    add_db_synapses(s, db_name, db_host, db_col_name + '_synapses',
                    selected_roi, nodes=nodes)

    add_layer(s, vanilla_seg, 'vanilla segmentation')
    add_layer(s, mtlsd_seg, 'mtlsd segmentation')
    add_layer(s, raw, 'raw')
    s.navigation.position.voxelCoordinates = np.flip(
        (selected_roi.get_begin() / raw[0].voxel_size))

print(viewer)
