import json
import os
import sys

predict = __import__('01_predict_blockwise')
watershed = __import__('02_extract_fragments_blockwise')
agglomerate = __import__('03_agglomerate_blockwise')
find_segments = __import__('04_find_segments')
extract_segmentation = __import__('04_extract_segmentation_from_lut')

if __name__ == '__main__':

    experiment = "hausser"
    setup = "setup01"
    iteration = 500000
    block_size = [5022,5022,5022]
    context = [930,620,620]
    num_workers = 3
    db_host = "mongodb://lsdAdmin:C20H25N3O@funke-mongodb3.int.janelia.org:27017/admin?replicaSet=rsLsd",

    path_to_configs = sys.argv[1]

    for i in range(7,9):

        roi_path = os.path.join(path_to_configs, 'sub_roi_%i.json'%i)

        print(roi_path)

        predict_config = {
              "experiment": experiment,
              "setup" : setup,
              "iteration" : iteration,
              "raw_file" : roi_path,
              "raw_dataset" : "train/raw_clahe/s0",
              "out_file" : "/nrs/funke/sheridana/hausser",
              "file_name": "sub_rois/neuron_test_%i.zarr"%i,
              "num_workers": num_workers,
              "db_host": db_host,
              "db_name": "hausser_test_neuron_%s_sub_roi_%i"%(setup,i),
              "queue": "slowpoke"
            }

        watershed_config = {
              "experiment": experiment,
              "setup": setup,
              "iteration": iteration,
              "affs_file": "/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "affs_dataset": "/volumes/affs",
              "fragments_file": "/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "fragments_dataset": "/volumes/fragments",
              "block_size": block_size,
              "context": context,
              "db_host": db_host,
              "db_name": "hausser_test_neuron_%s_sub_roi_%i"%(setup,i),
              "num_workers": num_workers,
              "fragments_in_xy": True,
              "queue": "normal"
            }

        agglomerate_config = {
              "experiment": experiment,
              "setup": setup,
              "iteration": iteration,
              "affs_file": "/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "affs_dataset": "/volumes/affs",
              "fragments_file": "/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "fragments_dataset": "/volumes/fragments",
              "block_size": block_size,
              "context": context,
              "db_host": db_host,
              "db_name": "hausser_test_neuron_%s_sub_roi_%i"%(setup,i),
              "num_workers": num_workers,
              "queue": "normal",
              "merge_function": "hist_quant_50"
            }

        find_segments_config = {
              "db_host": db_host,
              "db_name": "hausser_test_neuron_%s_sub_roi_%i"%(setup,i),
              "fragments_file": "/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "fragments_dataset": "/volumes/fragments",
              "edges_collection": "edges_hist_quant_50",
              "thresholds_minmax": [0.7,1],
              "thresholds_step": 1,
              "num_workers": num_workers
        }

        extract_segmentation_config = {
              "fragments_file":"/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "fragments_dataset": "/volumes/fragments",
              "edges_collection": "edges_hist_quant_50",
              "threshold": 0.7,
              "out_file": "/nrs/funke/sheridana/hausser/%s/%s/sub_rois/neuron_test_%i.zarr"%(setup,str(iteration),i),
              "out_dataset": "volumes/segmentation_70",
              "num_workers": num_workers
            }

        predict.predict_blockwise(**predict_config)
        watershed.extract_fragments(**watershed_config)
        agglomerate.agglomerate(**agglomerate_config)
        find_segments.find_segments(**find_segments_config)
        extract_segmentation.extract_segmentation(**extract_segmentation_config)
