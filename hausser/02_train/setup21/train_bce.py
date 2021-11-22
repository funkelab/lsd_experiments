from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Train
import gunpowder as gp
import math
import numpy as np
import torch
import logging
import os
import glob

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data/psds/2d/'

samples = glob.glob(os.path.abspath(data_dir+'*.zarr'))

# network parameters
num_fmaps = 12
input_shape = gp.Coordinate((96, 96))
output_shape = gp.Coordinate((56, 56))

batch_size = 32  # TODO: increase later

voxel_size = gp.Coordinate((62,62))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

label_proportions = [
        (0, 0.9960917319916444),
        (1, 0.003908268008355607)
]

def train(iterations):

    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=[
            (2, 2),
            (2, 2)],
        kernel_size_down=[
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]]],
        kernel_size_up=[
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]]])

    model = torch.nn.Sequential(
        unet,
        ConvPass(num_fmaps, 1, [[1, 1]],activation=None),
        torch.nn.Sigmoid())

    # loss = torch.nn.CrossEntropyLoss(
            # weight=torch.tensor([1 / p for l, p in label_proportions]))

    loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(1/0.004))

    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    predictions = gp.ArrayKey('PREDICTIONS')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(predictions, output_size)

    sources = tuple(
        gp.ZarrSource(
            os.path.join(data_dir,sample),
            {
                raw: 'raw',
                labels: 'labels_tmp',
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False),
            }) +
        gp.RandomLocation()
        for sample in samples
    )

    # raw:      (h, w)
    # labels:   (h, w)

    pipeline = sources
    pipeline += gp.RandomProvider()
    pipeline += gp.Normalize(raw)
    pipeline += gp.SimpleAugment()
    # pipeline += gp.ElasticAugment(
        # control_point_spacing=(64,64),
        # jitter_sigma=(5.0, 5.0),
        # rotation_interval=(0, math.pi/2))
    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    # add "channel" dimensions
    pipeline += gp.Unsqueeze([raw, labels])

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=10)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: predictions
        },
        loss_inputs={
            0: predictions,
            1: labels
        },
        save_every=10000)

    pipeline += gp.Snapshot({
            raw: 'raw',
            labels: 'labels',
            predictions: 'predictions'
        },
        every=500)

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)

if __name__ == '__main__':

    train(100000)
