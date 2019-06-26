import luigi

import sys
import os
sys.path.append('luigi_scripts')
import numpy as np
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'fib25',
        'setups': [
            'setup01',
            'setup03',
            'setup04',
            'setup06',
        ],
        'iterations': [400000],
        'samples': ['testing/fib25grid.json'],
        'gt_file': '/nrs/turaga/debd/fib25/3d_fib25_whitelisted_erode_r2_m50_mask_bineroded_eval.n5',
        'block_size': [2048, 2048, 2048],
        'context': [512, 512, 512],
        'fragments_in_xy': False,
        'epsilon_agglomerate': 0,
        'mask_fragments': True,
        'mask_file': '/nrs/turaga/debd/fib25/03_predict/setup04/400000/testing/fib25.n5',
        'merge_functions': [
            'hist_quant_75',
            'hist_quant_50',
            'mean',
            'hist_quant_50_initmax',
        ],
        'border_threshold': 0,
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
