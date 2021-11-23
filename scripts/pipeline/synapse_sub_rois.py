import daisy
import json
import os
import sys
import numpy as np

predict = __import__('01_predict_blockwise_tmp')
extract_fragments = __import__('02_extract_fragments_blockwise')
agglomerate = __import__('03_agglomerate_blockwise')
create_luts = __import__('04_find_segments')
segment = __import__('04_extract_segmentation_from_lut')

def get_roi(
        center,
        voxel_size,
        context,
        shape):

    center = daisy.Coordinate(center[::-1])

    offset = center - (shape / 2)

    roi = daisy.Roi(offset, shape)

    roi = roi.grow(context, context).snap_to_grid(voxel_size, mode='grow')

    return roi

def create_seg(
        experiment,
        lsds_setup,
        auto_setup,
        iteration,
        lsds_path,
        auto_path,
        affs_dataset,
        fragments_dataset,
        db_host,
        lsds_db_name,
        auto_db_name,
        block_size,
        context,
        raw,
        roi,
        crop_json,
        num_workers):

    roi_config = {
            'container': raw,
            'offset': roi.get_begin(),
            'size': roi.get_shape()
        }

    with open(crop_json, 'w') as f:
        json.dump(roi_config, f)

    predict_lsds_config = {
            'experiment': experiment,
            'setup': lsds_setup,
            'iteration': iteration,
            'raw_file': crop_json,
            'raw_dataset': raw_ds,
            'out_path': lsds_path,
            'num_workers': num_workers,
            'db_host': db_host,
            'db_name': lsds_db_name,
            'queue': 'slowpoke'
        }

    predict_auto_config = {
            'experiment': experiment,
            'setup': auto_setup,
            'iteration': iteration,
            'raw_file': crop_json,
            'raw_dataset': raw_ds,
            'auto_path': lsds_path,
            'auto_dataset': 'volumes/lsds',
            'out_path': auto_path,
            'num_workers': num_workers,
            'db_host': db_host,
            'db_name': auto_db_name,
            'queue': 'gpu_tesla_large'
        }

    watershed_config = {
            "experiment": experiment,
            "setup": auto_setup,
            "iteration": iteration,
            "affs_file": auto_path,
            "affs_dataset": affs_dataset,
            "fragments_file": auto_path,
            "fragments_dataset": fragments_dataset,
            "block_size": block_size,
            "context": context,
            "db_host": db_host,
            "db_name": auto_db_name,
            "num_workers": num_workers,
            "fragments_in_xy": True,
            "queue": 'normal',
            "filter_fragments": 0.05
    }

    agglomeration_config = {
            "experiment": experiment,
            "setup": auto_setup,
            "iteration": iteration,
            "affs_file": auto_path,
            "affs_dataset": affs_dataset,
            "fragments_file": auto_path,
            "fragments_dataset": fragments_dataset,
            "block_size": block_size,
            "context": context,
            "db_host": db_host,
            "db_name": auto_db_name,
            "num_workers": num_workers,
            "queue": 'normal',
            "merge_function": "hist_quant_50"
    }

    luts_config = {
            "db_host": db_host,
            "db_name": auto_db_name,
            "fragments_file": auto_path,
            "fragments_dataset": fragments_dataset,
            "edges_collection": "edges_hist_quant_50",
            "thresholds_minmax": [0.3,1],
            "thresholds_step": 1,
            "num_workers": num_workers
        }

    segment_config = {
            "fragments_file": auto_path,
            "fragments_dataset": fragments_dataset,
            "edges_collection": "edges_hist_quant_50",
            "threshold": 0.3,
            "out_file": auto_path,
            "out_dataset": "volumes/segmentation",
            "num_workers": 1
        }

    predict.predict_blockwise(**predict_lsds_config)
    predict.predict_blockwise(**predict_auto_config)

    os.remove(crop_json)

    extract_fragments.extract_fragments(**watershed_config)
    agglomerate.agglomerate(**agglomeration_config)
    create_luts.find_segments(**luts_config)
    segment.extract_segmentation(**segment_config)

if __name__ == '__main__':

    context = daisy.Coordinate((160,)*3)
    voxel_size = daisy.Coordinate((40,4,4))
    shape = daisy.Coordinate((3000,)*3)

    centers = [
            [482615, 264438, 89520],
            [581384, 316510, 64120],
            [372898, 149485, 163360],
            [688737, 161902, 132080],
            [638402, 135220, 189640],
            [816532, 256995, 188320],
            [555149, 156067, 116280],
            [399958, 277127, 131880],
            [678989, 222355, 92120],
            [315897, 355319, 203960],
            [458405, 190861, 103160]
            ]

    vols = [4,5,6,7,8,9,1,2,3,10,11]

    raw = '/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5'
    base_out_path = '/nrs/funke/sheridana/fafb_synapse_segs/'

    experiment = 'cremi'
    lsds_setup = 'setup177_p'
    auto_setup = 'setup189_p'
    iteration = 400000
    affs_dataset = 'volumes/affs'
    fragments_dataset = 'volumes/fragments'
    db_host = "mongodb://lsdAdmin:C20H25N3O@funke-mongodb3.int.janelia.org:27017/admin?replicaSet=rsLsd"
    db_name = 'fafb_synapse_segmentations_'
    raw_ds = 'volumes/raw'
    num_workers = 5
    block_size = [2240, 2240, 2240]
    post_context = [320, 256, 256]

    for c,v in zip(centers,vols):

        roi = get_roi(
                c,
                voxel_size,
                context,
                shape)

        v = str(v)

        print('Block %s roi: %s'%(v, roi))

        zarr_name = 'block_%s.zarr'%v

        lsds_path = os.path.join(
                base_out_path,
                lsds_setup,
                str(iteration),
                zarr_name)

        auto_path = os.path.join(
                base_out_path,
                auto_setup,
                str(iteration),
                zarr_name)

        lsds_db_name = db_name + '%s_%i_block_%s'%(lsds_setup,iteration,v)
        auto_db_name = db_name + '%s_%i_block_%s'%(auto_setup,iteration,v)

        crop_json = zarr_name.replace('zarr','json')

        create_seg(
            experiment=experiment,
            lsds_setup=lsds_setup,
            auto_setup=auto_setup,
            iteration=iteration,
            lsds_path=lsds_path,
            auto_path=auto_path,
            affs_dataset=affs_dataset,
            fragments_dataset=fragments_dataset,
            db_host=db_host,
            lsds_db_name=lsds_db_name,
            auto_db_name=auto_db_name,
            block_size=block_size,
            context=post_context,
            raw=raw,
            roi=roi,
            crop_json=crop_json,
            num_workers=num_workers)
