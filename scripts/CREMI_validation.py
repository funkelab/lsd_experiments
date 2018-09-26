import luigi

import sys
import os
sys.path.append('luigi_scripts')
import numpy as np
from tasks import *

if __name__ == '__main__':

    combinations = {
        'experiment': 'cremi',
        'setups': ['setup01'], #, 'setup04'],
        'iterations': [400000], #, 500000],
        'samples': ["testing/sample_C_padded_20160501.aligned.filled.cropped.62:153.n5"],
        'block_size':[2240, 2048, 2048],
        'context': [320, 256, 256],
        'num_workers': 40,
        'thresholds': list(np.arange(0,1,0.01)),
        'fragments_in_xy': True,
        'mask_fragments': True,
        'block_size': [2240, 2048, 2048],
        'context': [320, 256, 256],
        'border_threshold': 25
    }

    range_keys = [
        'setups',
        'iterations',
        'samples'
    ]

    set_base_dir(os.path.abspath('..'))

    luigi.build(
            [EvaluateCombinations(combinations, range_keys)],
            workers=50,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/funke/home/funkej/.luigi/logging.conf'
    )
