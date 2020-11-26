import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import optim
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import math
from sklearn.metrics import mean_squared_error
from read_data import read_data
from draw import draw

energy_path = '../data/energy_results_fpga/energy_data.pkl'


def build_dataset():
    X_full, Y = read_data(energy_path)
    X = list()
    for x in X_full:
        energy_data = list()
        for block in x:
            energy_data.append(block[0])
        X.append(energy_data)
    
    max_len = max([len(x) for x in X])
    for x in X:
        if len(x) == max_len: continue
        x.extend([0] * (max_len - len(x)) )
    
    return np.asarray(X, dtype=np.float), np.asarray(Y, dtype=np.float)



def train(model):
    X, Y = build_dataset()
    length = int(len(X) * 0.8)
    train_feature, train_y = X[:length], Y[:length]
    test_feature, test_y = X[length:], Y[length:]
    model.fit(train_feature, train_y)
    pred_y = model.predict(test_feature)
    # report rMSE
    mse = mean_squared_error(test_y, pred_y)
    rmse = math.sqrt(mse)     
    print(f'test set prediction rMSE = {rmse}')

    test_y_add = [sum(v) for v in test_feature]

    draw(test_y, pred_y, test_y_add, title='MLP (Energy)', path='mlp.pdf')

    



def mlp_main():
    clf = MLPRegressor(solver='adam',
                       alpha=1e-4,
                       hidden_layer_sizes=(100, 100, 100),
                       random_state=1,
                       max_iter=10000)
    train(clf)


if __name__ == "__main__":
    mlp_main()