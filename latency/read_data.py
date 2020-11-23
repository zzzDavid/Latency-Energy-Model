import json
import numpy as np
from glob import glob
import os

"""
    About the data format for latency:

    We sampled 2k subnets from OFA supernet, and each
    subnet was run 10 times. Each run result is saved in
    a json file, when we read in the json file it is a 
    dictionary. 
    Keys:
    (0)_firstconv_3x3_Conv_I3_O40_I(64,3,128,128)_O(64,40,64,64)
    (1)_(3x3_MBConv1_RELU6_I40_O24, None)_I(64,40,64,64)_O(64,24,64,64)
    (2)_(7x7_MBConv4_RELU6_I24_O32, None)_I(64,24,64,64)_O(64,32,32,32)
    (3)_(3x3_MBConv4_RELU6_I32_O32, Identity)_I(64,32,32,32)_O(64,32,32,32)
    (4)_(3x3_MBConv6_RELU6_I32_O32, Identity)_I(64,32,32,32)_O(64,32,32,32)
    (5)_(3x3_MBConv6_RELU6_I32_O56, None)_I(64,32,32,32)_O(64,56,16,16)
    (6)_(5x5_MBConv4_RELU6_I56_O56, Identity)_I(64,56,16,16)_O(64,56,16,16)
    (7)_(3x3_MBConv3_RELU6_I56_O104, None)_I(64,56,16,16)_O(64,104,8,8)
    (8)_(5x5_MBConv6_RELU6_I104_O104, Identity)_I(64,104,8,8)_O(64,104,8,8)
    (9)_(3x3_MBConv4_RELU6_I104_O104, Identity)_I(64,104,8,8)_O(64,104,8,8)
    (10)_(7x7_MBConv3_RELU6_I104_O104, Identity)_I(64,104,8,8)_O(64,104,8,8)
    (11)_(3x3_MBConv6_RELU6_I104_O128, None)_I(64,104,8,8)_O(64,128,8,8)
    (12)_(3x3_MBConv4_RELU6_I128_O128, Identity)_I(64,128,8,8)_O(64,128,8,8)
    (13)_(5x5_MBConv4_RELU6_I128_O248, None)_I(64,128,8,8)_O(64,248,4,4)
    (14)_(5x5_MBConv6_RELU6_I248_O248, Identity)_I(64,248,4,4)_O(64,248,4,4)
    (15)_(5x5_MBConv4_RELU6_I248_O416, None)_I(64,248,4,4)_O(64,416,4,4)
    (16)_feature_mix_layer_1x1_Conv_I416_O1664_I(64,416,4,4)_O(64,1664,4,4)
    block_latency_sum
    overall

    values: corresponding latency value
"""


def read_json(jsonFile_name):
    f = open(jsonFile_name)
    jsonData = json.load(f)
    f.close()
    return jsonData


"""
    data path should look like:
    ../data/latency_results_cpu_gpu/latency_table_b64_cpu
"""
def read_data(data_path):
    """
        return a list of dictionaries.
    """
    samples = os.listdir(data_path)
    data = list()
    for sample in samples:
        sample = os.path.join(data_path, sample)
        runs = glob(os.path.join(sample, '*.json'))
        if len(runs) == 0: continue
        datum = dict()

        block_latency_sum_avg = sum([read_json(run)['block_latency_sum'] for run in runs])
        block_latency_sum_avg /= len(runs)

        for run in runs:
            jsonData = read_json(run)
            block_latency_sum = jsonData['block_latency_sum']
            overall_latency = jsonData['overall']
            # 去除数据不正常或者波动过大的点
            if block_latency_sum < overall_latency: continue
            if abs(block_latency_sum - block_latency_sum_avg) > (block_latency_sum_avg * 0.1): continue
            for key, value in jsonData.items():
                if key in datum:
                    datum[key] += value
                else:
                    datum[key] = value
        if len(datum) == 0: continue
        # average over runs
        for key, value in datum.items():
            datum[key] = value / len(runs)
        # datum is ready, put it in the list
        data.append(datum)
    return data

