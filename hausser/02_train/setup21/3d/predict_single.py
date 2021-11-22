from funlib.learn.torch.models import UNet, ConvPass
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import logging
import zarr
import daisy

logging.basicConfig(level=logging.INFO)

num_fmaps = 12

grow= gp.Coordinate((36, 64, 64))

input_shape = gp.Coordinate((84,156,156)) + grow
output_shape = gp.Coordinate((36,64,64)) + grow

voxel_size = gp.Coordinate((93,62,62))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

print(input_size, output_size)

context = (input_size - output_size) / 2

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction*weights,
            target*weights)

def predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset):

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    source = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
    total_output_roi = total_input_roi.grow(-context, -context)

    print(total_input_roi, total_output_roi)

    daisy.prepare_ds(
            out_file,
            out_dataset,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.float32,
            write_size=output_size,
            num_channels=1)

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

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())

    checkpoint = torch.load(f'model_checkpoint_{iteration}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    predict = gp.torch.Predict(
            model,
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred: out_dataset
            },
            output_filename=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([raw, pred]) +
            gp.Squeeze([pred]) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred] = total_output_roi

    # predict_request.add(raw, total_input_roi.get_end())
    # predict_request.add(pred, total_output_roi.get_end())

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

if __name__ == "__main__":

    iteration = 500000
    raw_file = sys.argv[1]
    raw_dataset = 'train/raw_clahe/s0'
    out_file = sys.argv[2]
    out_dataset = 'volumes/sdt'

    predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset)
