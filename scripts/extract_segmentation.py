import json
import logging
import lsd
import os
import peach
import sys
import z5py
import numpy as np

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def extract_segmentation(
        experiment,
        setup,
        iteration,
        sample,
        threshold):

    experiment_dir = '../' + experiment
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    filename = os.path.join(predict_dir, sample)

    f = z5py.File(filename, mode='r+')

    # open fragments
    fragments_ds = f['volumes/fragments']
    offset = peach.Coordinate(fragments_ds.attrs['offset'][::-1])
    voxel_size = peach.Coordinate(fragments_ds.attrs['resolution'][::-1])
    # TODO: this is what we should do...
    # total_roi = peach.Roi(
        # offset,
        # voxel_size*fragments_ds.shape)
    # fragments = peach.Array(
        # fragments_ds,
        # total_roi,
        # voxel_size)

    # but for now we use voxel coordindates
    total_roi = peach.Roi(
        (0,)*len(voxel_size),
        fragments_ds.shape)
    fragments = peach.Array(
        fragments_ds,
        total_roi,
        (1, 1, 1))

    # open RAG DB
    rag_db = os.path.join(predict_dir, sample + '.db')
    rag_provider = lsd.persistence.SqliteRagProvider(rag_db, 'r')

    # slice
    print("Reading fragments and RAG in %s"%total_roi)
    fragments = fragments[total_roi]
    rag = rag_provider[total_roi.to_slices()]

    print("Number of fragments: %d"%(len(np.unique(fragments.data))))
    print("Number of nodes in RAG: %d"%(len(rag.nodes())))

    # create a segmentation
    print("Merging...")
    rag.get_segmentation(threshold, fragments.data)

    # store segmentation
    print("Writing segmentation...")
    ds = peach.prepare_ds(
        filename,
        'volumes/segmentation',
        total_roi*voxel_size + offset,
        voxel_size,
        fragments.data.dtype)
    ds.data[:] = fragments.data
    # ds = f.create_dataset(
        # 'volumes/segmentation',
        # data=fragments.data,
        # compression='gzip')
    # ds.attrs['offset'] = offset[::-1]
    # ds.attrs['resolution'] = voxel_size[::-1]
    # ds.attrs['threshold'] = threshold

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
