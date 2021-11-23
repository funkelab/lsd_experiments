import json
import os
import sys
segment = __import__('04_extract_segmentation_from_lut')

if __name__ == '__main__':

    rois = ['32']

    path_to_configs = sys.argv[1]

    setups = ['setup18', 'setup19', 'setup21', 'setup22', 'setup23', 'setup24', 'setup25']
    iterations = [400,200,200,400]
    db_names = ['mtlsd_padding','auto_basic_padding','auto_full_padding','mtlsd_malis_padding']
    ts = [0, 0, 0, 0]
    te = [1, 1, 1, 1]

    for i in rois:

        with open(os.path.join(path_to_configs, 'zebrafinch_%s_micron_roi.json'%i), 'r') as f:
            config = json.load(f)

        print(config)

        for a,b,c,d,e in zip(setups, iterations, db_names, ts, te):
            segments_config = {
                  "db_host": "mongodb://lsdAdmin:C20H25N3O@funke-mongodb3.int.janelia.org:27017/admin?replicaSet=rsLsd",
                  "db_name": "zebrafinch_%s_%ik_testing_masked_ffn"%(c,b),
                  "fragments_file": "/nrs/funke/sheridana/zebrafinch/%s/%i000/zebrafinch.zarr"%(a,b),
                  "edges_collection": "edges_hist_quant_50",
                  "thresholds_minmax": [float(f'{d}'), float(f'{e}')],
                  "thresholds_step": 0.02,
                  "roi_offset": config["roi_offset"],
                  "roi_shape": config["roi_shape"],
                  "run_type": "%s_micron_roi_masked_ffn"%i,
                  "num_workers": 32
            }

            print(segments_config)

            find_segments.find_segments(**segments_config)
