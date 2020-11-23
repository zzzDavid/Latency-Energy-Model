import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from read_data import read_data

latency_path = '../data/latency_results_cpu_gpu/latency_table_b64_cpu'


# prepare data
list_of_dict = read_data(latency_path)
df = dict()
df['block_latency_sum'] = list()
df['overall_latency'] = list()
df['block_num'] = list()
for datum in list_of_dict:
    df['block_latency_sum'].append(datum['block_latency_sum'])
    df['overall_latency'].append(datum['overall'])
    df['block_num'].append(len(datum) - 2)

train_x_len = round((len(df) / 3) * 2)


#单变量回归
x = df['block_latency_sum']
y = df['overall_latency']

train_x = np.array(x[1:train_x_len]) # a list of list
test_x = np.array(x[train_x_len+1:])
train_y = np.array(y[1:train_x_len])
test_y = np.array(y[train_x_len+1:])

train_x = train_x.reshape(-1,1)
test_x = test_x.reshape(-1,1)

model = linear_model.LinearRegression()
model.fit(train_x,train_y)

test_y_pred = model.predict(test_x)

pcc_sk = stats.pearsonr(test_y,test_y_pred)[0]
print('Single Factor: Pearson Correlation Coefficient(sklearn) = ' + str(pcc_sk))
r2score = r2_score(test_y,test_y_pred)
print('Single Factor: r2_score =' + str(r2score))
mse_ori = mean_squared_error(test_y,test_x)
mse_pred = mean_squared_error(test_y,test_y_pred)
print('Single Factor: original MSE =' + str(mse_ori) + ' predict MSE = ' + str(mse_pred))

t = np.arange(len(test_x))
plt.plot(t, test_y, 'r-', linewidth=2, label='真实数据')
plt.plot(t, test_y_pred, 'go-', linewidth=2, label='预测数据')
plt.savefig('1.pdf')

plt.plot(t, test_x, 'r-', linewidth=2, label='block_sum')
plt.plot(t, test_y, 'go-', linewidth=2, label='model_lat')
plt.savefig('2.pdf')

#多变量回归
x = [[a, b] for a, b in zip(df['block_num'], df['block_latency_sum'])]
y = df['overall_latency']

train_x = np.array(x[1:train_x_len]) # a list of list
test_x = np.array(x[train_x_len+1:])
train_y = np.array(y[1:train_x_len])
test_y = np.array(y[train_x_len+1:])

model = linear_model.LinearRegression()
model.fit(train_x,train_y)

test_y_pred = model.predict(test_x)

pcc_sk = stats.pearsonr(test_y,test_y_pred)[0]
print('Multiple Factors: Pearson Correlation Coefficient(sklearn) = ' + str(pcc_sk))
r2score = r2_score(test_y,test_y_pred)
print('Multiple Factors: r2_score =' + str(r2score))
mse_ori = mean_squared_error(test_y,test_x[:,1:])
mse_pred = mean_squared_error(test_y,test_y_pred)
print('Multiple Factors: original MSE =' + str(mse_ori) + ' predict MSE = ' + str(mse_pred))

# t = np.arange(len(test_x))
# plt.plot(t, test_y, 'r-', linewidth=2, label='真实数据')
# plt.plot(t, test_y_pred, 'go-', linewidth=2, label='预测数据')
# plt.show()

