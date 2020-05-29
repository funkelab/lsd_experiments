import daisy
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

def replace_in_block(
        block,
        in_ds,
        replace_dic,
        replaced_ds):

    # logging.info('Reading block in %s' %block.read_roi)
    replaced = in_ds.to_ndarray(block.write_roi)

    block_begin = block.read_roi.get_begin()

    shape = block.read_roi.get_shape()

    mapping = {}

    for i,j in zip(
            range(replaced.shape[0]),
            range(shape[0])
            ):
        mapping[i]=i
        mapping[j]=int((block_begin[0]/60)+i)

    r = [k for k,v in mapping.items() if v in replace_dic.keys()]
    w = [k for k,v in mapping.items() if v in replace_dic.values()]

    for i,j in zip(r,w):
        print('Replacing section %i with section %i'%(i,j))
        assert i != j, "section %i is equal to section %i"%(i,j)
        replaced[i] = replaced[j]

    replaced_ds[block.write_roi] = replaced

def replace(
        in_file,
        in_ds,
        out_file,
        out_ds,
        # roi_offset,
        # roi_shape,
        num_workers,
        replace_dic):

    logging.info('Loading data set...')

    in_ds = daisy.open_ds(in_file, in_ds)

    # total_roi = daisy.Roi((roi_offset), (roi_shape))

    total_roi = in_ds.roi

    read_roi = daisy.Roi((0, 0, 0), (18000, 16800, 16800))
    write_roi = read_roi

    logging.info('Creating cropped dataset...')

    replaced_ds = daisy.prepare_ds(
                    out_file,
                    out_ds,
                    total_roi,
                    in_ds.voxel_size,
                    dtype=in_ds.dtype,
                    write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: replace_in_block(
            b,
            in_ds,
            replace_dic,
            replaced_ds),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False)


if __name__ == '__main__':


    in_file = sys.argv[1]
    in_ds = 'volumes/labels/interpolated_mask_replaced_sections/s0'
    out_file = in_file
    out_ds = 'volumes/labels/delete'
    # roi_offset = [123420, 63784, 219408]
    # roi_shape = [31740, 31696, 31696]
    num_workers = 40

    #get sections to replace

    # replace_dic = {
            # 1767: 1768,
            # 1782: 1783,
            # 1890: 1891,
            # 1912: 1913,
            # 1992: 1993,
            # 2119: 2120,
            # 2145: 2146,
            # 2162: 2163,
            # 2173: 2174,
            # 2204: 2205,
            # 2295: 2296,
            # 2341: 2342,
            # 2427: 2428,
            # 2447: 2448,
            # 2501: 2502,
            # 2556: 2557,
            # 2602: 2603,
            # 2680: 2679,
            # 2687: 2688,
            # 2751: 2752,
            # 2805: 2806,
            # 2840: 2841,
            # 2869: 2870,
            # 2947: 2948,
            # 2976: 2977,
            # 3001: 3002,
            # 3016: 3017,
            # 3022: 3023,
            # 3102: 3103,
            # 3149: 3147,
            # 3150: 3147,
            # 3151: 3147,
            # 3152: 3147,
            # 3153: 3147,
            # 3154: 3147,
            # 3155: 3147,
            # 3156: 3147,
            # 3157: 3147,
            # 3158: 3147,
            # 3159: 3147,
            # 3160: 3147,
            # 3161: 3147,
            # 3162: 3147,
            # 3163: 3178,
            # 3164: 3178,
            # 3165: 3178,
            # 3166: 3178,
            # 3167: 3178,
            # 3168: 3178,
            # 3169: 3178,
            # 3170: 3178,
            # 3171: 3178,
            # 3172: 3178,
            # 3173: 3178,
            # 3174: 3178,
            # 3175: 3178,
            # 3176: 3178,
            # 3177: 3178,
            # 3256: 3257,
            # 3265: 3266,
            # 3279: 3280,
            # 3360: 3361,
            # 3394: 3395,
            # 3423: 3424,
            # 3259: 3460,
            # 3474: 3475,
            # 3476: 3477,
            # 3501: 3502,
            # 3505: 3506,
            # 3518: 3519,
            # 3523: 3524,
            # 3558: 3559,
            # 3691: 3692,
            # 3712: 3713,
            # 3723: 3724,
            # 3738: 3739,
            # 3794: 3795,
            # 3824: 3825,
            # 4148: 4149,
            # 4365: 4366,
            # 4374: 4375,
            # 4417: 4416,
            # 4419: 4420,
            # 4435: 4434,
            # 4437: 4438,
            # 4564: 4563,
            # 4566: 4567,
            # 4575: 4576,
            # 4652: 4653,
            # 4731: 4732,
            # 4742: 4743,
            # 4838: 4839,
            # 4937: 4938,
            # 4963: 4962,
            # 4964: 4962,
            # 4965: 4968,
            # 4966: 4968,
            # 5414: 5415,
            # 5549: 5550,
            # 5561: 5562,
            # 5579: 5580,
            # 5759: 5780,
            # 6062: 6063,
            # 6149: 6150,
            # 6152: 6151,
            # 6154: 6155,
            # 6262: 6263,
            # 6300: 6301,
            # 6319: 6320,
            # 6777: 6778,
            # 7246: 7247,
            # 7435: 7436,
            # 7524: 7525,
            # 7708: 7709,
            # 7747: 7748,
            # 7942: 7943,
            # 8028: 8029,
            # 9351: 9352,
            # 10523: 10522,
            # 10524: 10522,
            # 10525: 10522,
            # 10526: 10528,
            # 10527: 10528
            # }

    replace_dic = {
            547: 0,
            746: 0,
            1032: 0,
            1101: 0,
            1114: 0,
            2842: 0,
            3155: 0,
            3279: 0,
            3459: 0,
            4965: 0,
            4966: 0,
            7708: 7709,
            10524: 10523,
            10525: 10523,
            10526: 10528,
            10527: 10528
    }

    replace(
            in_file,
            in_ds,
            out_file,
            out_ds,
            # roi_offset,
            # roi_shape,
            num_workers,
            replace_dic)

