from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.torch import Train
import gunpowder as gp
import math
import numpy as np
import torch
import logging
import os
import glob
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from scipy.ndimage import generate_binary_structure

logging.basicConfig(level=logging.INFO)
# logging.getLogger('gp.gp.nodes.balance_labels').setLevel(logging.DEBUG)

data_dir = '../../01_data/psds/2d/'

samples = glob.glob(os.path.abspath(data_dir+'*.zarr'))

# network parameters
num_fmaps = 24
input_shape = gp.Coordinate((96, 96))
output_shape = gp.Coordinate((56, 56))

batch_size = 48  # TODO: increase later

voxel_size = gp.Coordinate((62,62))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
                prediction*weights,
                target*weights)

class DistanceTransform(gp.BatchFilter):

      def __init__(self, array, scale=None, constant=0.5):
          self.array = array

          if scale is not None:
              self.scale = scale
          else:
              self.scale = 1

          self.constant = constant

      def process(self, batch, request):

          data = batch[self.array].data.astype(np.uint8)

          inner = distance_transform_edt(binary_erosion(data))
          outer = distance_transform_edt(np.logical_not(data))

          sdt = (inner - outer) + self.constant

          scaled = np.clip(sdt, 0, 1).astype(np.float32)

          # scaled = np.tanh(sdt / self.scale).astype(np.float32)

          batch[self.array].data = scaled

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
            [[3, 3], [3, 3]]],
        constant_upsample=True)

    model = torch.nn.Sequential(
        unet,
        ConvPass(num_fmaps, 1, [[1, 1]],activation=None),
        torch.nn.Sigmoid())

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    predictions = gp.ArrayKey('PREDICTIONS')
    weights = gp.ArrayKey('WEIGHTS')
    gradients = gp.ArrayKey('GRADIENTS')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(predictions, output_size)
    request.add(weights, output_size)
    request.add(gradients, output_size)

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
    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    pipeline += gp.BalanceLabels(
            labels,
            weights)

    pipeline += DistanceTransform(labels)

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
            1: labels,
            2: weights
        },
        gradients={
            0:gradients
        },
        save_every=10000)

    pipeline += gp.Snapshot({
            raw: 'raw',
            labels: 'labels',
            predictions: 'predictions',
            gradients: 'gradients'
        },
        every=100,
        output_filename='batch_{iteration}.zarr')

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)

if __name__ == '__main__':

    train(20000)
