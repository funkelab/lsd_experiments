# convert existing h5 file to n5
# this only works if h5py is available
import os
import h5py
import z5py
import numpy as np
from z5py import File
from itertools import product
from concurrent import futures


def blocking(shape, block_shape):
    """ Generator for nd blocking.
    """
    if len(shape) != len(block_shape):
        raise RuntimeError("Invalid number of dimensions.")
    ranges = [range(sha // bsha if sha % bsha == 0 else sha // bsha + 1)
              for sha, bsha in zip(shape, block_shape)]
    start_points = product(*ranges)
    for start_point in start_points:
        positions = [sp * bshape for sp, bshape in zip(start_point, block_shape)]
        yield tuple(slice(pos, min(pos + bsha, sha))
                    for pos, bsha, sha in zip(positions, block_shape, shape))


def convert_from_h5(in_path,
                        out_path,
                        in_path_in_file,
                        out_path_in_file,
                        out_chunks,
                        n_threads,
                        out_blocks=None,
                        use_zarr_format=None,
                        **z5_kwargs):
        """ Convert hdf5 dataset to n5 or zarr dataset.
        The chunks of the output dataset must be spcified.
        The dataset is converted in parallel over the chunks.
        Datatype and compression can be specified, otherwise defaults will be used.
        Args:
            in_path (str): path to hdf5 file.
            out_path (str): path to output zarr or n5 file.
            in_path_in_file (str): name of input dataset.
            out_path_in_file (str): name of output dataset.
            out_chunks (tuple): chunks of output dataset.
            n_threads (int): number of threads used for converting.
            out_blocks (tuple): block size used for converting, must be multiple of ``out_chunks``.
                If None, the chunk size will be used (default: None).
            use_zarr_format (bool): flag to indicate zarr format.
                If None, an attempt will be made to infer the format from the file extension,
                otherwise zarr will be used (default: None).
            **z5_kwargs: keyword arguments for ``z5py`` dataset, e.g. datatype or compression.
        """
        if not os.path.exists(in_path):
            raise RuntimeError("Path %s does not exist" % in_path)
        if out_blocks is None:
            out_blocks = out_chunks

        f_z5 = File(out_path, use_zarr_format=use_zarr_format)
        with h5py.File(in_path, 'r') as f_h5:
            ds_h5 = f_h5[in_path_in_file]
            shape = ds_h5.shape

            # modify z5 arguments
            out_dtype = z5_kwargs.pop('dtype', ds_h5.dtype)
            if 'compression' not in z5_kwargs:
                z5_kwargs['compression'] = 'raw'
            ds_z5 = f_z5.create_dataset(out_path_in_file,
                                        dtype=out_dtype,
                                        shape=shape,
                                        chunks=out_chunks,
                                        **z5_kwargs)

            def convert_chunk(bb):
                # print("Converting chunk ", chunk_ids, "/", chunks_per_dim)
                ds_z5[bb] = ds_h5[bb].astype(out_dtype, copy=False)

            with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
                tasks = [tp.submit(convert_chunk, bb)
                         for bb in blocking(shape, out_blocks)]
                [t.result() for t in tasks]

            # copy attributes
            h5_attrs = ds_h5.attrs
            z5_attrs = ds_z5.attrs
            for key, val in h5_attrs.items():
		z5_attrs[key] = val

h5_file = '../fib25/01_data/testing/cube01_gt.hdf'
n5_file = '../fib25/01_data/testing/cube01.n5'
h5_key = n5_key = 'volumes/labels/neuron_ids'
target_chunks = (64, 64, 64)
n_threads = 8

convert_from_h5(h5_file, n5_file,
        in_path_in_file=h5_key,
        out_path_in_file=n5_key,
        out_chunks=target_chunks,
        n_threads=n_threads,
        compression='gzip')
