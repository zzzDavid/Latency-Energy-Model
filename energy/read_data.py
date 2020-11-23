import pickle
import os


def read_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X = list()
    Y = list()
    for net, data_dict in data.items():
        X.append(data_dict['block_energy'])
        Y.append(data_dict['overall_energy'])
    return X, Y