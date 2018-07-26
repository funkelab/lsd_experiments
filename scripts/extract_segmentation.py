import logging
import lsd
import peach
import z5py

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.sqlite_rag_provider').setLevel(logging.DEBUG)

out_file = 'cube04.n5'
fragments_ds = 'volumes/fragments'
rag_db = 'cube04.db'

roi = peach.Roi((0, 0, 0), (1024, 1024, 1024))

if __name__ == "__main__":

    f = z5py.File(out_file, mode='r+')

    # open fragments
    fragments = f[fragments_ds]
    voxel_size = peach.Coordinate(fragments.attrs['resolution'])
    offset = peach.Coordinate(fragments.attrs['offset'])

    # open RAG DB
    rag_provider = lsd.persistence.SqliteRagProvider(rag_db, 'r')

    # slice
    print("Reading fragments and RAG...")
    fragments = fragments[roi.to_slices()]
    rag = rag_provider[roi.to_slices()]

    # create a segmentation
    print("Merging...")
    rag.get_segmentation(0.55, fragments)

    # store segmentation
    print("Writing segmentation...")
    ds = f.create_dataset(
        'volumes/segmentation',
        data=fragments,
        compression='gzip')
    ds.attrs['offset'] = roi.get_offset()*voxel_size + offset
    ds.attrs['resolution'] = voxel_size
