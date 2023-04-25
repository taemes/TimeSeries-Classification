
import os
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay


start = time.time()
seed = 41
torch.manual_seed(seed)
np.random.seed(seed)

# File Load #
path_train, path_test = './data/', './data/'
file_train, file_test = 'train_data_2.csv', 'test_data.csv'

Train = os.path.join(path_train, file_train)
Test = os.path.join(path_test, file_test)
dataTrain = np.loadtxt(Train, delimiter=',', dtype=np.str_)
dataTest = np.loadtxt(Test, delimiter=',', dtype=np.str_)


XTrain = np.float32(dataTrain[:, 2:-1])
YTrain = np.float32(dataTrain[:, [-1]])

XTest = np.float32(dataTest[1:, 2:-1])
YTest = np.float32(dataTest[1:, [-1]])

classes = ('Rock', 'Scissor', 'Paper')


# Data Augmentation #
def augmentation(x, factor):
    noise = np.random.normal(loc=0, scale=factor, size=x.shape)

    s_factor = np.random.normal(loc=1.0, scale=factor, size=(1, x.shape[1]))
    scale = np.matmul(np.ones((x.shape[0], 1)), s_factor)
    shift = np.matmul(np.ones((x.shape[0], 1)), s_factor)

    noise, scale = x + noise, x * scale
    shift_u, shift_d = x + shift, x - shift

    aug_x = np.concatenate([noise, scale, shift_u, shift_d])
    aug_y = YTrain
    for _ in range(3):
        aug_y = np.vstack((aug_y, YTrain))

    return aug_x, aug_y


aug_x1, aug_y1 = augmentation(torch.Tensor(XTrain), 0.1)
aug_x2, aug_y2 = augmentation(torch.Tensor(XTrain), 0.15)

XTrain = np.vstack((XTrain, aug_x1, aug_x2))
YTrain = np.vstack((YTrain, aug_y1, aug_y2))

XTrain_tensors = Variable(torch.FloatTensor(XTrain))
YTrain_tensors = Variable(torch.LongTensor(YTrain))
XTrain_tensors_final = torch.reshape(XTrain_tensors,
                                     (XTrain_tensors.shape[0], 1, XTrain_tensors.shape[1]))


# device 선정 #
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 사용
device = torch.device('cpu')  # CPU 사용


# 네트워크 파라미터 구성 #
batch_size = 128  # 2의 거듭제곱으로 설정
num_epochs = 3000
learning_rate = 1e-1

input_size = 5              # number of features
hidden_size = 2             # number of features in hidden state
num_layers = 1              # number of stacked lstm layers
num_classes = len(classes)  # number of output classes

dataset = TensorDataset(XTrain_tensors_final, YTrain_tensors)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# LSTM 네트워크 구성 #
class LstmClass(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LstmClass, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers    # number of layers
        self.input_size = input_size    # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length    # seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)  # LSTM
        self.fc_1 = nn.Linear(hidden_size, 124)  # fully connected 1
        self.fc = nn.Linear(124, num_classes)    # fully connected last layer
        self.fc_2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        out = self.fc_2(hn)
        out = self.act(out)

        return out


lstm = LstmClass(num_classes, input_size, hidden_size, num_layers, XTrain_tensors_final.shape[1]).to(device)
loss_function = torch.nn.CrossEntropyLoss()                       # loss function: CrossEntropy
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)  # optimizer: SGD

y_one_hot = torch.zeros(len(YTrain_tensors), len(classes))
y_one_hot.scatter_(1, YTrain_tensors, 1)  # one-hot encoding


# training #
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(data_loader):

        outputs = lstm.forward(XTrain_tensors_final.to(device))
        loss = F.cross_entropy(outputs, y_one_hot.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# prediction #
df_x_nz = XTest
df_x_nz = Variable(torch.FloatTensor(df_x_nz))
df_x_nz = torch.reshape(df_x_nz, (df_x_nz.shape[0], 1, df_x_nz.shape[1]))
df_y_nz = Variable(torch.LongTensor(YTest))

TestSet = TensorDataset(df_x_nz, df_y_nz)
TestLoader = DataLoader(TestSet, batch_size=batch_size, shuffle=False)


# accuracy #
correct = {ClassName: 0 for ClassName in classes}
total = {ClassName: 0 for ClassName in classes}
with torch.no_grad():
    for test in TestLoader:
        data, labels = test
        train_predict = lstm(df_x_nz.to(device))
        _, predictions = torch.max(train_predict, 1)
        predictions = predictions.data.detach().cpu().numpy()

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct[classes[int(label)]] += 1
            total[classes[int(label)]] += 1


# 학습 시간 #
end = time.time()
sec = end - start
process_time = str(datetime.timedelta(seconds=sec)).split(".")
print(process_time[0])


# Confusion Matrix #
cm = confusion_matrix(df_y_nz, predictions, normalize='true')
cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot()
plt.show()

# pip install matplotlib==3.2.0


