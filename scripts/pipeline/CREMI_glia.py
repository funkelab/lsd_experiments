import luigi

import sys
sys.path.append('luigi_scripts')
from tasks import *

if __name__ == '__main__':

    luigi.build(
            [
                SegmentTask(
                    experiment='cremi',
                    setup='setup55_g',
                    iteration=400000,
                    sample='testing/sample_%s+repeat.n5'%s,
                    block_size=[2240, 2048, 2048],
                    context=[320, 256, 256],
                    fragments_in_xy=True,
                    epsilon_agglomerate=0,
                    mask_fragments=True,
                    merge_function='hist_quant_75_initmax',
                    threshold=0.3)
                for s in ['A', 'B', 'C']
            ],
            # [
                # SegmentTask(
                    # experiment='cremi',
                    # setup='setup56_g',
                    # iteration=300000,
                    # sample='testing/sample_%s+black.n5'%s,
                    # block_size=[2240, 2048, 2048],
                    # context=[320, 256, 256],
                    # fragments_in_xy=True,
                    # epsilon_agglomerate=0,
                    # mask_fragments=True,
                    # merge_function='hist_quant_75_initmax',
                    # threshold=0.3)
                # for s in ['A', 'B', 'C']
            # ] +
            # [
                # SegmentTask(
                    # experiment='cremi',
                    # setup='setup57_g',
                    # iteration=300000,
                    # sample='testing/sample_%s+black.n5'%s,
                    # block_size=[2240, 2048, 2048],
                    # context=[320, 256, 256],
                    # fragments_in_xy=True,
                    # epsilon_agglomerate=0,
                    # mask_fragments=True,
                    # merge_function='hist_quant_75_initmax',
                    # threshold=0.3)
                # for s in ['A', 'B', 'C']
            # ],
            
            workers=20,
            scheduler_host='slowpoke1.int.janelia.org',
            logging_conf_file='/groups/funke/home/funkej/.luigi/logging.conf'
    )
