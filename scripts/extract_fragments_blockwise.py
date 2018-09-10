import json
import logging
import lsd
import numpy as np
import os
import daisy
import sys

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def extract_fragments(
        experiment,
        setup,
        iteration,
        sample,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
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

        fragments_in_xy (``bool``):

            Extract fragments section-wise.

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
    affs = daisy.open_ds(in_file, affs_ds, mode='r')

    if mask_fragments:

        data_dir = os.path.join(experiment_dir, '01_data')
        sample_file = os.path.abspath(os.path.join(data_dir, sample))
        logging.info("Reading mask from %s", sample_file)
        mask = daisy.open_ds(sample_file, 'volumes/labels/mask', mode='r')

    else:

        mask = None

    # prepare fragments dataset
    fragments = daisy.prepare_ds(
        out_file,
        fragments_ds,
        affs.roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0, 0, 0), block_size))

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='w')
    logging.info("RAG DB opened")

    # extract fragments in parallel
    lsd.parallel_watershed(
        affs,
        rag_provider,
        block_size,
        context,
        fragments,
        num_workers,
        fragments_in_xy,
        mask=mask)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_fragments(**config)
