import h5py
import z5py

print('[WARNING] will overwrite dataset')
with h5py.File('predictions2.hdf', 'w') as f:
    ds = f.create_dataset('volumes/affs', shape=(3, 1240, 1240, 1240), dtype='float32')
    ds.attrs['resolution'] = (8, 8, 8)
    ds.attrs['offset'] = (114000, 39400, 96800)
    ds.attrs['origin'] = 'fib19.n5/volumes/raw-aligned/s0'
