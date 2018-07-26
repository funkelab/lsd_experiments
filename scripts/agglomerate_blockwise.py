import json
import logging
import lsd
import numpy as np
import os
import sys
import z5py

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def agglomerate(
        experiment,
        setup,
        iteration,
        sample,
        block_size,
        context,
        num_workers,
        retry):
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        sample (``string``):

            Name of the sample to predict in, relative to the experiment's data
            dir. Should be an HDF5 or N5 container with 'volumes/raw'.

        block_size (``tuple`` of ``int``):

            The size of one block in voxels.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration.

        num_workers (``int``):

            How many blocks to run in parallel.

        retry (``int``):

            How many times to retry failed tasks.
    '''

    experiment_dir = '../' + experiment
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    in_file = os.path.join(predict_dir, sample)
    affs_ds = 'volumes/affs'
    out_file = in_file
    fragments_ds = 'volumes/fragments'
    rag_db = os.path.join(predict_dir, sample + '.db')

    if not os.path.isdir(os.path.join(in_file, affs_ds)):
        raise RuntimeError(
            "No affinity predictions found for %s, %s, %s"%(
                experiment, setup, iteration))

    print("Reding affs from %s"%in_file)

    # open affs
    affs = z5py.File(in_file, use_zarr_format=False, mode='r')[affs_ds]

    print("Read affs with shape %s"%(affs.shape,))

    # open or create fragments dataset
    if not os.path.isdir(os.path.join(out_file, fragments_ds)):

        print("Creating new fragments dataset in %s"%out_file)

        with z5py.File(out_file, use_zarr_format=False, mode='r+') as f:
            fragments = f.create_dataset(
                fragments_ds,
                shape=affs.shape[1:],
                chunks=block_size,
                dtype=np.uint64,
                compression='gzip')
            fragments.attrs['offset'] = affs.attrs['offset']
            fragments.attrs['resolution'] = affs.attrs['resolution']

    else:

        print("Re-using existing fragments dataset in %s"%out_file)

        # check if output dataset is what we expect
        fragments = z5py.File(out_file, use_zarr_format=False, mode='r+')[fragments_ds]

        assert fragments.shape==affs.shape[1:], (
            "Existing fragments dataset has different shape than input "
            "volume.")
        assert tuple(fragments.chunks)==tuple(block_size), (
            "Existing fragments dataset has chunk size %s different from "
            "%s"%(fragments.chunks, block_size,))

    # open RAG DB
    rag_provider = lsd.persistence.SqliteRagProvider(rag_db, 'r+')

    # extract fragments in parallel
    for i in range(retry + 1):

        if lsd.parallel_watershed(
            affs,
            rag_provider,
            block_size,
            context,
            fragments,
            num_workers):
            break

        if i < retry:
            print("parallel_watershed failed, retrying %d/%d"%(i + 1, retry))

    # agglomerate in parallel
    for i in range(retry + 1):

        if lsd.parallel_aff_agglomerate(
            affs,
            fragments,
            rag_provider,
            block_size,
            context,
            merge_function='OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
            threshold=1.0,
            num_workers=num_workers):
            break

        if i < retry:
            print("parallel_aff_agglomerate failed, retrying %d/%d"%(i + 1, retry))

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    agglomerate(**config)