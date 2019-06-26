import os
import subprocess
import numpy as np
import zarr
import h5py
import sys
import time
import glob
import json
import threading
from vol2mesh.mesh_from_array import mesh_from_array
import multiprocessing as mp

def load_ds(segmentation_file, segmentation_dataset, mode='r'):

    print('Loading segmentation into memory...')

    try:
        if segmentation_file.endswith('zarr') or segmentation_file.endswith('n5'):
            ds = zarr.open(segmentation_file, mode=mode)[segmentation_dataset][:]
        else:
            ds = h5py.open(segmentation_file, mode=mode)[segmentation_dataset][:]

    except KeyError:
        print('Dataset %s could not be loaded' %ds)
        return None

    print('Succesfully loaded segmentation')

    return ds

def check_if_complete(out_dir):

    complete = glob.glob(os.path.join(out_dir, "*.obj"))

    return complete


def generate_single_mesh(
        mask,
        box,
        downsample_factor,
        simplify_ratio,
        smoothing_rounds,
        output_format,
        out_dir,
        l
        ):

    print('Creating Mesh from array')

    mesh = mesh_from_array(
            mask,
            box,
            downsample_factor,
            simplify_ratio,
            smoothing_rounds,
            output_format)

    obj = os.path.join(out_dir, 'mesh_%i.obj'%l)

    with open(obj, 'wb') as f:
        f.write(mesh)

def generate_meshes(
        segmentation_file,
        segmentation_ds,
        out_dir,
        downsample_factor,
        simplify_ratio,
        smoothing_rounds,
        output_format):

    ds = load_ds(segmentation_file, segmentation_ds)

    print('Creating unique ids for neurons...')
    labels = np.unique(ds)

    procs = []

    os.makedirs(out_dir, exist_ok=True)

    for l in labels:

        if l == 0:
            continue
        if 'mesh_%i.obj'%l in check_if_complete(out_dir):
            print('Already generated mesh for id %i, skipping' %l)
            continue

        mask = ds == l
        box = [(0,0,0), (mask.shape)]

        proc = mp.Process(
                target=generate_single_mesh,
                args=(
                    mask,
                    box,
                    downsample_factor,
                    simplify_ratio,
                    smoothing_rounds,
                    output_format,
                    out_dir,
                    l))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    generate_meshes(**config)
    print('Ran script in %.3f seconds'%(time.time() - start))
