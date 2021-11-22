import daisy
import gunpowder as gp
import json
import logging
import math
import numpy as np
import os
import pymongo
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)
# logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    net_config = json.load(f)

num_fmaps = 12

input_shape = gp.Coordinate(net_config['input_shape'])
output_shape = gp.Coordinate(net_config['output_shape'])

voxel_size = gp.Coordinate((93,62,62))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

context = (input_size - output_size) / 2

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction*weights,
            target*weights)

unet = UNet(
    in_channels=1,
    num_fmaps=num_fmaps,
    fmap_inc_factor=5,
    downsample_factors=[
        (1,2,2),
        (2,2,2),
        (2,2,2)])

model = torch.nn.Sequential(
    unet,
    ConvPass(num_fmaps, 1, [[1, 1, 1]],activation=None),
    torch.nn.Sigmoid())

def block_done_callback(
        db_host,
        db_name,
        worker_config,
        block,
        start,
        duration):

    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    document = dict(worker_config)
    document.update({
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    })

    collection.insert(document)

    print("Recorded block-done for %s" % (block,))

def predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset,
        db_host,
        db_name,
        worker_config,
        model,
        **kwargs):

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    chunk_request = gp.BatchRequest()

    chunk_request.add(raw, input_size)
    chunk_request.add(pred, output_size)

    pipeline = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True)
            })

    pipeline += gp.Pad(raw, size=None)

    pipeline += gp.Normalize(raw)
    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Unsqueeze([raw])
    # pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.torch.Predict(
            model,
            checkpoint=os.path.join(
                setup_dir,
                f'model_checkpoint_{iteration}'),
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred
            })

    pipeline += gp.Squeeze([pred])
    pipeline += gp.Squeeze([pred])

    pipeline += gp.ZarrWrite(
            dataset_names={
                pred: out_dataset
            },
            output_filename=out_file)

    pipeline += gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
                pred: 'write_roi'
            },
            num_workers=worker_config['num_cache_workers'],
            block_done_callback=lambda b, s, d: block_done_callback(
                db_host,
                db_name,
                worker_config,
                b, s, d))

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    # iteration = 450000
    # raw_file = sys.argv[1]
    # raw_dataset = 'raw'
    # out_file = raw_file
    # out_dataset = 'pred_450k'

    # predict(
        # iteration,
        # raw_file,
        # raw_dataset,
        # out_file,
        # out_dataset)


    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    iteration = run_config['iteration']

    model.eval()

    run_config['model'] = model

    predict(**run_config)
