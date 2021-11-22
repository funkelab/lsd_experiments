import copy
import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
from funlib.learn.torch.models import UNet
from typing import List

torch.backends.cudnn.benchmark = True

class ToDtype(gp.BatchFilter):
    """ Cast arrays to another numerical datatype

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys for typecasting
        dtype: output data type as string
        output_arrays (List[gp.ArrayKey]): optional, ArrayKeys for outputs
    """

    def __init__(self,
                 arrays: List[gp.ArrayKey],
                 dtype,
                 output_arrays: List[gp.ArrayKey] = None):
        self.arrays = arrays
        self.dtype = dtype

        if output_arrays:
            assert len(arrays) == len(output_arrays)
        self.output_arrays = output_arrays

    def setup(self):
        self.enable_autoskip()

        if self.output_arrays:
            for in_array, out_array in zip(self.arrays, self.output_arrays):
                if not out_array:
                    raise NotImplementedError(
                        'Provide no output_arrays or one for each input_array')
                else:
                    self.provides(out_array, self.spec[in_array].copy())
        else:
            for array in self.arrays:
                self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()

        if self.output_arrays:
            output_arrays = self.output_arrays
        else:
            output_arrays = self.arrays

        for in_array, out_array in zip(self.arrays, output_arrays):
            deps[in_array] = request[out_array].copy()

        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        if self.output_arrays:
            output_arrays = self.output_arrays
        else:
            output_arrays = self.arrays

        for in_array, out_array in zip(self.arrays, output_arrays):
            outputs[out_array] = copy.deepcopy(batch[in_array])
            outputs[out_array].spec.dtype = self.dtype
            outputs[out_array].data = batch[in_array].data.astype(self.dtype)

        return outputs

logging.basicConfig(level=logging.INFO)

voxel_size = gp.Coordinate((93,62,62))

input_shape = gp.Coordinate((84,156,156))
output_shape = gp.Coordinate((36,64,64))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

context = (input_size - output_size) / 2

num_fmaps = 12
batch_size = 1

base_dir = '../../01_data/'
samples = ['test_volume.zarr', 'new_mitos.zarr']

def calc_context(voxel_size, output_size):

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = gp.Roi(
                    (gp.Coordinate(
                        [i/2 for i in [output_size[0], diag, diag]]) +
                        voxel_size),
                    (0,)*3).snap_to_grid(voxel_size,mode='shrink')

    return max_padding.get_begin()

context = calc_context(voxel_size, output_size)

label_proportions = [
    (0, 0.81724853515625),
    (1, 0.18275146484375)
]

unet = UNet(
    in_channels=1,
    num_fmaps=num_fmaps,
    fmap_inc_factor=5,
    downsample_factors=[(1, 2, 2), (1, 2, 2)],
    constant_upsample=True)

model = torch.nn.Sequential(
        unet,
        torch.nn.Conv3d(num_fmaps, 2, (1,) * 3))

loss = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([1 / p for l, p in label_proportions]))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

raw = gp.ArrayKey("RAW")
labels = gp.ArrayKey("LABELS")
pred = gp.ArrayKey("PRED")
grad = gp.ArrayKey("GRAD")

request = gp.BatchRequest()
request.add(raw, input_size)
request.add(labels, output_size)
request.add(pred, output_size)
request.add(grad, output_size)

sources = tuple(gp.ZarrSource(
            os.path.join(base_dir, sample),
            datasets={
                raw:'raw',
                labels:'labels_2'
            },
            array_specs={
                raw:gp.ArraySpec(interpolatable=True),
                labels:gp.ArraySpec(interpolatable=False)
            }) +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.Pad(labels, context) +
            gp.RandomLocation()
            for sample in samples)

pipeline = sources

pipeline += gp.RandomProvider()

pipeline += gp.ElasticAugment(
        control_point_spacing=(30, 520, 520),
        jitter_sigma=(0, 52, 52),
        rotation_interval=(0, math.pi / 2))

pipeline += gp.SimpleAugment(transpose_only=[1, 2])

pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)

# pytorch expects b, c, d, h, w (5d tensor)

# raw: d, h, w

pipeline += gp.Unsqueeze([raw])

# raw: 1, d, h, w (c,d,h,w)

pipeline += gp.Stack(batch_size)

# raw: 1, 1, d, h, w (b,c,d,h,w)

pipeline += gp.PreCache(
            cache_size=40,
            num_workers=10)

pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        outputs={0: pred},
        loss_inputs={
            0: pred,
            1: labels
        },
        gradients={0: grad},
        save_every=5000,
        log_dir='log')

pipeline += ToDtype([labels], dtype=np.uint64)

pipeline += gp.Squeeze([raw, labels, pred, grad])

pipeline += gp.Snapshot(
        output_filename="batch_{iteration}.zarr",
        dataset_names={
            raw: "raw",
            labels: "labels",
            pred: "pred",
            grad: "grad"
        },
        every=500)

if __name__ == "__main__":
    with gp.build(pipeline):
        for i in range(int(sys.argv[1])):
            batch = pipeline.request_batch(request)
