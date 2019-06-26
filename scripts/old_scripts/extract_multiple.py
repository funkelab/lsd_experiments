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
        thresholds):

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

    for threshold in thresholds:
        
        segmentation = fragments.data.copy()

        # create a segmentation
        print("Merging...")
        rag.get_segmentation(threshold, segmentation)

        # store segmentation
        print("Writing segmentation for threshold %f" % (threshold))
        seg = daisy.prepare_ds(
            filename,
            'volumes/segmentation/' + str(threshold),
            fragments.roi,
            fragments.voxel_size,
            fragments.data.dtype)
        seg.data[:] = segmentation.data

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
