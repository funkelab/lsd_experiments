import h5py
import z5py
import peach

def get_dataset(filename, dataset):

    if filename.endswith('h5') or filename.endswith('hdf'):

        ds = h5py.File(filename)[dataset]
        if 'offset' in ds.attrs:
            offset = ds.attrs['offset']
        else:
            offset = [0, 0, 0]
        if 'resolution' in ds.attrs:
            resolution = ds.attrs['resolution']
        else:
            resolution = [1, 1, 1]
        return (
            ds,
            peach.Coordinate(offset),
            peach.Coordinate(resolution)
        )

    elif filename.endswith('n5'):

        ds = z5py.File(filename)[dataset]
        if 'offset' in ds.attrs:
            offset = ds.attrs['offset']
        else:
            offset = [0, 0, 0]
        if 'resolution' in ds.attrs:
            resolution = ds.attrs['resolution']
        else:
            resolution = [1, 1, 1]
        return (
            ds,
            peach.Coordinate(offset[::-1]),
            peach.Coordinate(resolution[::-1])
        )

    else:
        raise RuntimeError("Unknown file format for %s"%filename)
