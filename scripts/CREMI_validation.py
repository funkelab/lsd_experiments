import luigi

import sys
import os
sys.path.append('luigi_scripts')
import numpy as np
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'cremi',
        'setups': [
            #'setup01',
            #'setup03',
            #'setup04',
            'setup04_f',
            #'setup05',
            'setup05_f',
            #'setup07',
            #'setup08',
            'setup08_f',
            #'setup09',
            #'setup12',
            #'setup22',
            #'setup23',
            #'setup28',
            #'setup34',
            #'setup35',
            #'setup40',
            #'setup41',
            #'setup46',
            #'setup47'
        ],
        'iterations': [300000, 400000, 500000],
        'samples': ['testing/sample_C_padded_20160501.aligned.filled.cropped.62:153.n5'],
        'block_size': [2240, 2048, 2048],
        'context': [320, 256, 256],
        'fragments_in_xy': True,
        'epsilon_agglomerate': 0,
        'mask_fragments': True,
        'merge_functions': [
            'hist_quant_25_initmax',
            'hist_quant_50_initmax',
            'hist_quant_75_initmax',
            'mean'
        ],
        'border_threshold': 25,
        'thresholds_minmax': [0, 1],
        'thresholds_step': 0.01
    }

    range_keys = [
        'setups',
        'iterations',
        'samples',
        'merge_functions'
    ]

    luigi.build(
            [EvaluateCombinations(combinations, range_keys)],
            workers=20,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/funke/home/funkej/.luigi/logging.conf'
    )
