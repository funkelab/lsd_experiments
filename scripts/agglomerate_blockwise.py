import json
import logging
import lsd
import numpy as np
import os
import daisy
import sys
#import config

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def agglomerate(
        experiment,
        setup,
        iteration,
        in_file,
        affs_dataset,
        fragments_dataset,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
        merge_function):
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        experiment (``string``):

            Name of the experiment (cremi, fib19, fib25, ...).

        setup (``string``):

            Name of the setup to predict.

        iteration (``int``):

            Training iteration to predict from.

        in_file (``string``):

            The input file containing affs and fragments.

        affs_dataset, fragments_dataset (``string``):

            Where to find the affinities and fragments.

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
    '''

    experiment_dir = '../' + experiment
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    logging.info("Reading affs from %s", in_file)
    affs = daisy.open_ds(in_file, affs_dataset, mode='r')

    if mask_fragments:
        
        mask_ds = 'volumes/labels/mask'
        logging.info("Reading mask from %s", in_file)
        mask = daisy.open_ds(in_file, mask_ds, mode='r')

    logging.info("Reading fragments from %s", in_file)
    fragments = daisy.open_ds(in_file, fragments_dataset, mode='r')

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r+')
    logging.info("RAG DB opened")

    # agglomerate in parallel

    lsd.parallel_aff_agglomerate(
        affs,
        fragments,
        rag_provider,
        block_size,
        context,
        merge_function=merge_function,
        threshold=1.0,
        num_workers=num_workers)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    agglomerate(**config)
