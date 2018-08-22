import json
import logging
import lsd
import numpy as np
import os
import peach
import sys

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def agglomerate(
        experiment,
        setup,
        iteration,
        sample,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
        retry,
        fragments_in_xy=False,
        mask_fragments=False):
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

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.

        retry (``int``):

            How many times to retry failed tasks.

        mask_fragments (``bool``):

            Whether to mask fragments for a specified region. Requires that the
            original sample dataset contains a dataset ``volumes/labels/mask``.
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

    logging.info("Reading affs from %s", in_file)
    affs = peach.open_ds(in_file, affs_ds, mode='r')

    if mask_fragments:

<<<<<<< HEAD
        sample_file = os.path.abspath(os.path.join(data_dir, sample))
        print("Reading mask from %s"%sample_file)

        f = z5py.File(sample_file)
        if 'volumes/labels/mask' not in f:
            raise RuntimeError(
                "Masking requested, but no 'volumes/labels/mask' found in %s"%
                sample_file)
        mask = f['volumes/labels/mask']

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
=======
        logging.info("Reading mask from %s", sample_file)
        data_dir = os.path.join(experiment_dir, '01_data')
        sample_file = os.path.abspath(os.path.join(data_dir, sample))
        mask = peach.open_ds(sample_file, 'volumes/mask', mode='r')
>>>>>>> 2eeb677574484420cf04d4ade5a667c5938abf53

    else:

        mask = None

    # prepare fragments dataset
    fragments = peach.prepare_ds(
        out_file,
        fragments_ds,
        affs.roi,
        affs.voxel_size,
        np.uint64,
        peach.Roi((0, 0, 0), block_size))

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='w')
    logging.info("RAG DB opened")

    # extract fragments in parallel
    for i in range(retry + 1):

        if lsd.parallel_watershed(
            affs,
            rag_provider,
            block_size,
            context,
            fragments,
            num_workers,
            fragments_in_xy,
            mask=mask):
            break

        if i < retry:
            logging.error("parallel_watershed failed, retrying %d/%d", i + 1, retry)

    # agglomerate in parallel
    for i in range(retry + 1):

        if lsd.parallel_aff_agglomerate(
            affs,
            fragments,
            rag_provider,
            block_size,
            context,
            merge_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
            threshold=1.0,
            num_workers=num_workers):
            break

        if i < retry:
            logging.error("parallel_aff_agglomerate failed, retrying %d/%d", i + 1, retry)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    agglomerate(**config)
