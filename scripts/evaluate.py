import json
import logging
import lsd
import os
import daisy
import sys
import cremi

logging.basicConfig(level=logging.INFO)

def evaluate(
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

    # TODO:
    # * read gt (similar to fragments)
    # * create CremiVolume for gt
    # * create cremi.evaluation.NeuronIds with gt

    for threshold in thresholds:

        segmentation = fragments.data.copy()

        # create a segmentation
        print("Creating segmentation for threshold %f..."%threshold)
        rag.get_segmentation(threshold, segmentation)

        # TODO:
        # * create CremiVolume for segmentation
        # * get VOI and RAND from NeuronIds
        # * store values (maybe in DB?)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    extract_segmentation(**config)
