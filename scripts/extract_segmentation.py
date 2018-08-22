import json
import logging
import lsd
import os
import daisy
import sys

logging.basicConfig(level=logging.INFO)
# logging.getLogger('lsd.persistence.mongodb_rag_provider').setLevel(logging.DEBUG)

def extract_segmentation(
        experiment,
        setup,
        iteration,
        sample,
        db_host,
        db_name,
        threshold):

    experiment_dir = '../' + experiment
    predict_dir = os.path.join(
        experiment_dir,
        '03_predict',
        setup,
        str(iteration))

    filename = os.path.join(predict_dir, sample)

    # open fragments
    fragments = daisy.open_ds(filename, 'volumes/fragments')

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name,
        host=db_host,
        mode='r')

    total_roi = fragments.roi

    # slice
    print("Reading fragments and RAG in %s"%total_roi)
    fragments = fragments[total_roi]
    rag = rag_provider[total_roi]

    print("Number of nodes in RAG: %d"%(len(rag.nodes())))
    print("Number of edges in RAG: %d"%(len(rag.edges())))

    # create a segmentation
    print("Merging...")
    rag.get_segmentation(threshold, fragments.data)

    # store segmentation
    print("Writing segmentation...")
    segmentation = daisy.prepare_ds(
        filename,
        'volumes/segmentation',
        fragments.roi,
        fragments.voxel_size,
        fragments.data.dtype)
    segmentation.data[:] = fragments.data

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
