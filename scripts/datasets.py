import h5py
import z5py
import peach

def get_dataset(filename, dataset):

    if filename.endswith('h5') or filename.endswith('hdf'):

        ds = h5py.File(filename)[dataset]
        return (
            ds,
            peach.Coordinate(ds.attrs['offset']),
            peach.Coordinate(ds.attrs['resolution'])
        )

    elif filename.endswith('n5'):

        ds = z5py.File(filename)[dataset]
        return (
            ds,
            peach.Coordinate(ds.attrs['offset'][::-1]),
            peach.Coordinate(ds.attrs['resolution'][::-1])
        )

    else:
        raise RuntimeError("Unknown file format for %s"%filename)
