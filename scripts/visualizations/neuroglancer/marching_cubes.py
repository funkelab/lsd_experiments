import daisy
import logging
import numpy as np
import os
import sys
from skimage import measure
import json

from synful import database, synapse

logging.basicConfig(level=logging.INFO)


def generate_meshes(ids, name=None):
    mask = np.zeros_like(ds.data, dtype=bool)
    for i in ids:
        print(i)
        mask[ds == i] = True
    print(np.sum(mask), 'size of object')
    verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.5,
                                                                   spacing=voxel_size)

    faces = faces + 1
    if name is None:
        save_name = os.path.join(sys.argv[3], 'mesh_%i.obj' % (i))
    else:
        save_name = os.path.join(sys.argv[3], 'mesh_{}.obj'.format(name))

    obj = open(save_name, mode='w')

    for item in verts:
        obj.write("v {0} {1} {2}\n".format(item[2], item[1], item[0]))

    for item in normals:
        obj.write("vn {0} {1} {2}\n".format(item[2], item[1], item[0]))

    for item in faces:
        obj.write(
            "f {0}//{0} {1}//{1} {2}//{2}\n".format(item[2], item[1], item[0]))

    obj.close()


if __name__ == '__main__':

    logging.info('Loading file...')

    ds = daisy.open_ds(sys.argv[1], sys.argv[2])
    res_level = sys.argv[2].split('/')[-1]
    crop_to_roi = True # Assumes that a high res mesh is needed.

    voxel_size = ds.voxel_size
    print(ds.voxel_size, 'voxel_size')
    offset = ds.roi.get_begin()
    print('original ROI: {}'.format(ds.roi))

    print('outputdirectorys {}'.format(sys.argv[3]))

    new_roi = None
    # ROI for synapse complete high resolution

    # ROI for high resolution pair of neurons
    # center point= daisy.Roi((107608, 34264, 5059))
    if crop_to_roi:
        # new_roi = daisy.Roi(
        #     (5000 * 40 - 300 * 40, 34200 * 4 - 1000 * 4, 107600 * 4 - 1000 * 4),
        #     (600 * 40, 2100 * 4, 1700 * 4))
        center_point = np.array((108440, 34730, 5066)) # Neuroglancer copy paste
        center_point = center_point[::-1]
        center_point *= np.array((40, 4, 4))
        new_roi = daisy.Roi(center_point-np.array((10000, 10000, 10000)), (20000, 20000, 20000))
        new_roi = new_roi.snap_to_grid(ds.voxel_size)
        ds.roi = new_roi
    print(ds.roi, "ROI")
    print(ds.roi / ds.voxel_size)

    ds = ds.to_ndarray()
    print(ds.data.shape)

    if not os.path.exists(sys.argv[3]):
        os.mkdir(sys.argv[3])

    config_eval = sys.argv[4]
    with open(config_eval) as f:
        config_eval = json.load(f)

    # Pair v2
    # post_skel_id = 22906
    post_skel_id = 11146

    # Pair v1
    # post_skel_id = 8238
    # post_skel_id = 32214
    gt_db_host = config_eval['gt_db_host']
    gt_db_name = config_eval['gt_db_name']
    gt_db_col = config_eval['gt_db_col']
    print(gt_db_host, gt_db_name, gt_db_col)

    gt_db = database.NeuronDatabase(gt_db_name, db_host=gt_db_host,
                                    db_col_name=gt_db_col,
                                    mode='r')
    nodes = gt_db.read_neuron(post_skel_id)[0]
    print('number of nodes', len(nodes))
    ids = []
    for node in nodes:
        # print(node)
        ids.append(node['seg_id'])
    if crop_to_roi:
        new_ids = [id for id in ids if ids.count(id) > 1]
    else:
        new_ids = [id for id in ids if ids.count(id) > 1]
    new_ids = list(np.unique(new_ids))
    if 0 in new_ids:
        new_ids.remove(0)
    if post_skel_id == 8238:
        new_ids.remove(79543555)
        new_ids.remove(79339143)
    if post_skel_id == 11146:
        new_ids.remove(79339143)
        new_ids.append(133876124)
        new_ids.append(133832846)
        new_ids.append(136440276)
        new_ids.append(133995148)

    # if post_skel_id == 11146:
    #     new_ids = [169741673, 120592431]
    print(np.unique(new_ids))

    # generate_meshes(ids=np.unique(new_ids),
                    # name='{}_big_{}_cropped_{}'.format(post_skel_id, res_level,
                                                       # crop_to_roi))
