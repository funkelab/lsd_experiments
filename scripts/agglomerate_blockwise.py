import z5py
import logging
import lsd
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

def agglomerate(
        in_file,
        affs_ds,
        out_file,
        fragments_ds,
        rag_db,
        block_size,
        context,
        num_workers,
        retry):

    # open affs
    affs = z5py.File(in_file, mode='r')[affs_ds]

    # open or create fragments dataset
    if not os.path.isdir(os.path.join(out_file, fragments_ds)):

        print("Creating new fragments dataset...")
        with z5py.File(out_file, mode='w') as f:
            fragments = f.create_dataset(
                fragments_ds,
                shape=affs.shape[1:],
                chunks=block_size,
                dtype=np.uint64,
                compression='gzip')
            fragments.attrs['offset'] = affs.attrs['offset']
            fragments.attrs['resolution'] = affs.attrs['resolution']

    else:

        # check if output dataset is what we expect
        fragments = z5py.File(out_file, mode='r+')[fragments_ds]

        assert fragments.shape==affs.shape[1:], (
            "Existing fragments dataset has different shape than input "
            "volume.")
        assert fragments.chunks==block_size, (
            "Existing fragments dataset has chunk size different from "
            "%s"%(block_size,))

        print("Re-using existing fragments dataset")

    # open RAG DB
    rag_provider = lsd.persistence.SqliteRagProvider(rag_db, 'r+')

    # # extract fragments in parallel
    # for i in range(retry + 1):

        # if lsd.parallel_watershed(
            # affs,
            # rag_provider,
            # block_size,
            # context,
            # fragments,
            # num_workers):
            # break

        # if i < retry:
            # print("parallel_watershed failed, retrying %d/%d"%(i + 1, retry))

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

    agglomerate(
        '../03_predict/cube04.n5',
        'volumes/affs',
        'cube04.n5',
        'volumes/fragments',
        'cube04.db',
        block_size=(128, 128, 128),
        context=(8, 8, 8),
        num_workers=50,
        retry=2)
