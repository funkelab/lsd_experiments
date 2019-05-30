import glob
import itertools
import logging
import os
import sys

import daisy
import neuroglancer
import numpy as np

from synful import database

from funlib.show.neuroglancer import add_layer

neuroglancer.set_server_bind_address('0.0.0.0')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

ngid = itertools.count(start=1)

def add_db_synapses(s, db_name, db_host, db_col_name, source_roi, score_thr=0):
    dag_db = database.DAGDatabase(db_name, db_host, db_col_name=db_col_name,
                                  mode='r')
    edges = dag_db.read_edges(source_roi)
    nodes = dag_db.read_nodes_based_on_edges(edges)
    pos_dic = {}
    scores = []
    for node in nodes:
        score = node['score']
        scores.append(score)
        if score >= score_thr:
            node_id = node['id']
            pos = np.array(node['position'])
            pos_dic[node_id] = np.flip(pos)
    print('mean {0:.2f} and median {1:.2f} of score'.format(np.mean(scores), np.median(scores)))
    print('filtered out {} of synapses'.format(len(scores)-len(pos_dic)))


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
                                                              radii=(30, 30, 30),
                                                              id=next(ngid)))
            post_sites.append(neuroglancer.EllipsoidAnnotation(center=post_site,
                                                               radii=(30, 30, 30),
                                                               id=next(ngid)))
            connectors.append(
                neuroglancer.LineAnnotation(point_a=pre_site, point_b=post_site,
                                            id=next(ngid)))

    s.layers['connetors'] = neuroglancer.AnnotationLayer(
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


def open_ds_wrapper(path, ds_name):
    """Returns None if ds_name does not exists """
    try:
        return daisy.open_ds(path, ds_name)
    except KeyError:
        print('dataset %s could not be loaded' % ds_name)
        return None


if __name__ == '__main__':
    """
    Script for the visualization of block2, synaptic partners,
    synaptic clefts and pre_dist and post_dist.
    """

    raw_file = '/nrs/funke/buhmannj/data/block2/block2.hdf'
    inferencepath = '/nrs/funke/buhmannj/calyx/inference/' \
                    '2018setup02/240000/block2.zarr'
    voxel_size = (40, 4, 4)

    raw = open_ds_wrapper(raw_file, 'filtered/gray')
    pred_indicator = open_ds_wrapper(inferencepath,
                                     'volumes/pred_syn_indicator')

    # N5 Voxel Size is not loaded, set it manually and correct ROI.
    cleft_path = '/nrs/saalfeld/lauritzen/02/workspace.n5'
    cleft_df_cc = 'syncleft_dist_DTU-2_200000_cc'
    clefts_cc = open_ds_wrapper(cleft_path, cleft_df_cc)
    clefts_cc.voxel_size = voxel_size
    clefts_cc.roi = clefts_cc.roi * clefts_cc.voxel_size

    pre_dist = 'predictions_it150000_pre_and_post-v2.0/pre_dist/s0'
    pre_dist = open_ds_wrapper(cleft_path, pre_dist)
    pre_dist.voxel_size = voxel_size
    pre_dist.roi = pre_dist.roi * pre_dist.voxel_size

    post_dist = 'predictions_it150000_pre_and_post-v2.0/post_dist/s0'
    post_dist = open_ds_wrapper(cleft_path, post_dist)
    post_dist.voxel_size = voxel_size
    post_dist.roi = post_dist.roi * post_dist.voxel_size

    # Synaptic partner are stored in mongodb.
    db_name = 'synful_block2'
    db_host = 'c04u01'
    db_col_name = 'synapses'

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add_db_synapses(s, db_name, db_host, db_col_name,
                                      pred_indicator.roi, score_thr=50)
        add_layer(s, raw, 'raw')
        add_layer(s, clefts_cc, 'clefts')

        add_layer(s, pre_dist, 'pre_dist', shader='rgba', opacity=1.0, h=[0.3, 0, 0.3])
        add_layer(s, post_dist, 'post_dist', shader='rgba', opacity=1.0, h=[0, 0.1, 0.9])
        add_layer(s, pred_indicator, 'pred_post', shader='rgba', opacity=1.0, h=[0.1, 0.2, 0.2])

        s.navigation.position.voxelCoordinates = np.flip(
            (pred_indicator.roi.get_begin() / pred_indicator.voxel_size))

    print(viewer.__str__().replace('c04u01.int.janelia.org', '10.40.4.51'))
