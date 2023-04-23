# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:50:53 2022

@author: Ensyuan
"""

#---------------------------------------
# Mofan RNN-classifier
#---------------------------------------

# torch.manual_seed(1)    # reproducible

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Hyper Paramaters

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01
DOWNLOAD_MNIST = True


# MNIST dataset

train_data = torchvision.datasets.MNIST(root = './mnist/',
                                       train=True,
                                       transform = transforms.ToTensor(),   # Converts a PIL.Image or numpy.ndarray to
                                                                            # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                                       download = DOWNLOAD_MNIST)

test_data = torchvision.datasets.MNIST(root = './mnist/',
                                       train=False,
                                       transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

#test_loader = torch.utils.data.DataLoader(dataset=train_data, 
#                                          batch_size=BATCH_SIZE,
#                                          shuffle=False)

# convert test data into Variable, pick 2000 samples to speed up testing
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.    # shape (2000, 28, 28) value in range(0,1)
 # test data 先轉形態、取兩千個、再normalize
test_y = test_data.test_labels.numpy()[:2000]   # covert to numpy array


# plot an example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0]) # number
# = plt.title(f'{train_data.train_labels[0]}')
# = plt.title(train_data.train_labels[0].item())
plt.show()

# cf
plt.imshow(train_data.train_data[1].numpy(), cmap='gray')
plt.title(train_data.train_labels[1]) # tensor(number)
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = 64,
            num_layers=1,
            batch_first=True    # the first dimensioninput of input & output will be batch size . e.g. (batch, time_step, input_size)
            )
        
        self.out = nn.Linear(64, 10)
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        # x shape (batch, time_step, input_size) --> (64, 28, 28)
        # r_out shape (batch, time_step, output_size) --> (64, 28, 64)
        # h_n shape (n_layers, batch, hidden_size) # 副軸 hidden layer
        # h_c shape (n_layers, batch, hidden_size) # 主軸 hidden layer
        
        out = self.out(r_out[:, -1, :]) # 取最後一個time step 的結果
        return out
    
rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)
        
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy() # [1]: torch.max(test_output, 1)的第0項記錄values, 第一項記錄index
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            
# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy() # [1]: torch.max(test_output, 1)的第0項記錄values, 第一項記錄index
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
        
        
















