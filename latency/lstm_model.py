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
import os
from draw import draw

latency_path = '../data/latency_results_cpu_gpu/latency_table_b16_cuda'



def build_dataset():
    list_of_dict = read_data(latency_path)
    X = list()
    Y = list()
    for datum in list_of_dict:
        Y.append(datum['overall'])
        # let's build an input feature
        x_feature = list()
        for key, value in datum.items():
            if key == 'block_latency_sum' or key == 'overall': continue
            latency = value
            if 'firstconv' in key or 'feature' in key:
                kernel_size = 1
            else:
                kernel_size = int(key.split('_')[1].split('x')[1])
            input_dims = [int(v) for v in key.split('_')[-2].replace('I(','').replace(')','').split(',')]
            output_dims = [int(v) for v in key.split('_')[-1].replace('O(','').replace(')','').split(',')]
            x_feature.append([latency, kernel_size, *input_dims])
        X.append(x_feature)
    return X, Y




class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, device="cpu"):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True).to(self.device)
        self.linear = nn.Linear(hidden_layer_size, output_size).to(self.device)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
        

    def forward(self, seq, bs=1):
        self.hidden_cell = (torch.zeros(1, bs, self.hidden_layer_size).to(self.device),
                                    torch.zeros(1, bs, self.hidden_layer_size).to(self.device))
        seq = rnn_utils.pack_sequence([torch.tensor(s).reshape(-1, 1).to(self.device) for s in seq], enforce_sorted=False)
        lstm_out, self.hidden_cell = self.lstm(seq, self.hidden_cell)
        lstm_out, index = rnn_utils.pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.permute([1, 0, 2])
        select = torch.zeros(lstm_out.shape[:2]).scatter_(1, index.reshape(-1, 1) - 1, 1).to(torch.bool).to(self.device)
        lstm_out = lstm_out[select]
        predictions = self.linear(lstm_out)
        return predictions[:, -1]

    def predict(self, input_seqs):
        return self.forward(input_seqs, bs=len(input_seqs)).cpu().detach().numpy()

    def fit(self, train_X, train_y, epochs=200, bs=128):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for i in range(epochs):
            for j in range(len(train_X) // bs + 1):
                seq = train_X[j * bs: (j + 1) * bs]
                label = train_y[j * bs: (j + 1) * bs]
                optimizer.zero_grad()
                pred = self.forward(seq, bs=bs)
                loss = loss_function(pred, torch.tensor(label).to(pred.device))
                loss.backward()
                optimizer.step()

            if i % 5 == 0:
                print("epoch: {:3}, loss: {:10.5f}".format(
                    i, loss.item()))
        return self


def train(model, epoch=1000):
    X, Y = build_dataset()
    length = int(len(X) * 0.8)
    train_feature, train_y = X[:length], Y[:length]
    test_feature, test_y = X[length:], Y[length:]
    if os.path.exists('./lstm_weights.pth'):
        model.load_state_dict(torch.load('./lstm_weights.pth'))
    model.fit(train_feature, train_y, epochs=epoch)
    pred_y = model.predict(test_feature)
    # report rMSE
    mse = mean_squared_error(test_y, pred_y)
    rmse = math.sqrt(mse)     
    print(f'test set prediction rMSE = {rmse}')
    torch.save(model.state_dict(), './lstm_weights.pth')
    print("saved weights")


def test(model):
    model.load_state_dict(torch.load('./lstm_weights.pth'))
    print("loaded weights")

    X, Y = build_dataset()
    
    length = int((len(X) * 0.8))
    test_y = list()
    test_y_pred = list()
    simple_sum = list()
    for x, gt_y in zip(X[length:], Y[length:]):
        pred_y = model.predict([x])
        print(f"predicted = {pred_y}, measured = {gt_y}")
        test_y_pred.append(pred_y[0])
        test_y.append(gt_y)
        simple_sum.append(sum( [b[0] for b in x] ))
    # report rMSE
    mse = mean_squared_error(test_y, test_y_pred)
    rmse = math.sqrt(mse)    
    print(f'prediction rMSE = {rmse}')
    mse = mean_squared_error(test_y, simple_sum)
    rmse = math.sqrt(mse) 
    print(f'directly add rMSE = {rmse}')
    
    draw(test_y, test_y_pred, simple_sum, title='LSTM (Latency)', path='./lstm.pdf')


def lstm_main():
    model = LSTM(1, 20, 1, device="cuda")
    # train(model, epoch=100)
    test(model)

if __name__ == "__main__":
    lstm_main()
