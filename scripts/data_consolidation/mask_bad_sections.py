import zarr

# includes missing sections
bad_sections = {
    'A+': {
        'replace': [123, 122, 94, 93, 65, 47, 14, 2], # 125 and 29 somewhat bad, but okay
        'with'   : [124, 121, 95, 92, 66, 48, 15, 3]
    },
    'B+': {
        'replace': [91, 59, 58, 30, 29, 1], # 88 and 77 somewhat bad
        'with'   : [92, 60, 57, 31, 28, 2]
    },
    'C+': {
        'replace': [100, 88, 28],
        'with'   : [101, 89, 29]
    }
}

if __name__ == "__main__":

    for sample in bad_sections.keys():

        ds_raw = zarr.open('sample_%s_black.n5'%sample, mode='r+')['volumes/raw']

        for r, w in zip(
                bad_sections[sample]['replace'],
                bad_sections[sample]['with']):
            # ds_raw[r] = ds_raw[w]
            ds_raw[r] = 0
