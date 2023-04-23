# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:25:42 2022

@author: Ensyuan
"""

"""
# Anaconda prompt #
# 建置環境 
conda create -n pytorch-tutorial python=3.8
conda activate pytorch-tutorial
pip list # 確認環境中的package

# 安裝 pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip list # 確認環境中的package，現在包含torch
"""

# Spyder
import torch

#--------------------
# 02 Tensor Basic
#--------------------
# 隨機tensor
x = torch.empty(2,3) # e-39~e-38 的小數
print(x)
print(x.dtype)
print(x.size())

x = torch.rand(2,3) # 0-1之間的小數
print(x)

x = torch.zeros(2,3)
print(x)

x = torch.ones(2,3)
print(x)

x = torch.ones(2,3, dtype=torch.int)
print(x)
print(x.size())

# 給定tensor
x = torch.tensor([2.5, 3.1])
print(x)

# tensor 加減乘除
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z1 = x + y
print(z1)
z2 = torch.add(x,y)
print(z2)

y.add_(x) # a function with underscore will do an implace operation
print(y)  # so y = y + x

z3 = torch.sub(x,y) # y.sub_(x)
z4 = torch.mul(x,y) # y.mul_(x)
z5 = torch.div(x,y) # y.div_(x)

# 從tensor 取數字
x = torch.rand(5,3)
print(x)
print(x[:, 0])
print(x[1, :])

print(x[1,1])
print(x[1,1].item()) # 直接把數值印出來

# 改size
x = torch.rand(4,4)
print(x)
y = x.view(-1, 8) # -1 表示它會幫你找最適大小
print(y.size())
y = x.view(2, 8)
print(y.size())

# numpy 跟 torch 
import numpy as np
import torch
a = torch.ones(5)
print(a)
print(type(a))
b = a.numpy() # tensor 轉numpy
print(b)
print(type(b))

a.add_(1)
print(a)
print(b) # 注意b也+1了

c = np.ones(5)
print(c)
d = torch.from_numpy(c) # numpy 轉tensor
print(d)

c += 1
print(c)
print(d) # 注意d也+1了

torch.cuda.is_available() # False
if torch.cuda.is_available(): # if true
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu") # 存在GPU的tensor不能轉numpy, 需先傳回cpu
    z = z.numpy() 


x = torch.ones(5, requires_grad=True) # tell pytorch we'll need to calculate the gradient for this tensor later optimal session step 
print(x)

# -----------------------------
# 03 Autograd
# -----------------------------

x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2
z = z.mean() # 最後一步通常是sum()|mean()，使它成為 scalar value # A scalar is a simple single numeric value (as in 1, 2/3, 3.14, etc.)
# scalar value 量 c.f. vector 向量
print(z)
z.backward() # dz/dx
print(x.grad)


# vector jacobian  00:31:57
z = y*y*2 # 當 z 是陣列(向量)，而非一數(量) # 需要給一個vector
v = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) 
z.backward(v) # 這步驟無法重複
print(x.grad)


# 取消grdient
x = torch.rand(3, requires_grad=True)
print(x)
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad() 00:37:10

x.requires_grad_(False) # a function with underscore will do an implace operation
print(x) # x has no gradient

y = x.detach() # y has no gradient

with torch.no_grad():
    y = x+2  # y has no gradient
    print(y) 


# grad要清零，否則會疊加
weights = torch.ones(4, requires_grad=True)
 
for epoch in range(3):
     
    model_output = (weights*3).sum()
    
    model_output.backward()
    
    print(weights.grad)
    
    weights.grad.zero_() # grad要清零，否則會疊加


# ----------------------------- 
# 04 Backpropagation
# -----------------------------

# Chain Rule：dz/dx = dz/dy * dy/dx
# y_hat = w * x
# s = y_hat - y
# Loss = ( y_hat - y )^2 = ( wx - y )^2
# 要算 dLoss/dx，就要先算 dLoss/dz、dz/dx，再用Chain Rule
# 要 Minimise Loss 所以要算 dLoss/dw, weight 是要修正的參數 # important

# Step 1：Forward pass
# Step 2：Local Gradient (每一步的微分)
# Step 3：Backward pass

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# Step 1：forward pass and compute the loss
y_hat = w*x
loss = (y_hat - y)**2

print(loss)

# Step 3：Backward pass
loss.backward()
print(w.grad)


# -----------------------------
# 05 Gradient Descent
# -----------------------------

# 1) 用 numpy 做 learning
import numpy as np
# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction：to predict the w
def forward(x):
    return w * x

# loss = MSE mean square error
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x * (w*x - y) # MSE 對 w 微分 = dLoss/dw
# we need to calculate this differentiation by ourselves.
def gradient(x,y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Pridiction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20 # iteration

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients
    dw = gradient(X, Y, y_pred)
    
    # update weight
    w -= learning_rate * dw # dw是負數
    
    if epoch % 1 ==0:
        print (f'epoch {epoch+1}: dw = {dw:.3f}, w = {w:.3f}, loss = {l:.8f}')

print(f'Pridiction after training: f(5) = {forward(5):.3f}')

# 2) 用 pytorch 做 learning
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction：to predict the w
def forward(x):
    return w * x

# loss = MSE mean square error
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x * (w*x - y) # MSE 對 w 微分 = dLoss/dw
# def gradient(x,y, y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()
# NOW we DON'T need to calculate this differentiation by ourselves.

print(f'Pridiction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100 # iteration

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weight
    with torch.no_grad():
        w -= learning_rate * w.grad # dw是負數
    print(f'w_grad: {w.grad}')
    # zero gradients
    w.grad.zero_() # 將 dw 歸零
    
    if (epoch+1) % 10 ==0:
        print (f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Pridiction after training: f(5) = {forward(5):.3f}')


# -----------------------------
# 06 Training Pipeline
# -----------------------------

# 1) Design model (input, output, size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#     - Step 1：Forward pass
#     - Step 2：Backward pass: Gradient
#     - Step 3：update weight

import torch
import torch.nn as nn # neuronetwork

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32) # 需二維以上
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)  

input_size = n_features
output_size = n_features           

# model prediction：to predict the w
 # model = nn.Linear(input_size, output_size) # 不用自己寫 forward pass # 但這model不太準

# if you want a custom model, rewrite with this below. 01:24:50
 # 這個重寫的 model 我不太懂
class LinearRegression(nn.Module): # he said rewrite from the Module
    
    def __init__(self, input_dim, output_dim): # dim means dimension 
        super(LinearRegression, self).__init__()  #this is how we call the super constructor # super是？
        # define layers
        self.lin = nn.Linear(input_dim, output_dim) # linear layer
        
    def forward(self, x):
        return self.lin(x)
    
    # Here is just a dummy example which does the same with nn.Linear.
    
model = LinearRegression(input_size, output_size)

# loss = MSE mean square error
 # def loss(y, y_predicted):
     # return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x * (w*x - y) # MSE 對 w 微分 = dLoss/dw
 # def gradient(x,y, y_predicted):
 #     return np.dot(2*x, y_predicted-y).mean()

print(f'Pridiction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 200 # iteration

loss = nn.MSELoss() # 用內建的loss function，需要多維度
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD stands for statistic gradient decent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw 
    
    # update weight
    optimizer.step() # optimize the w
    
    # zero gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 10 ==0:
        [w, b] = model.parameters() # unpack the parameters
        print (f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Pridiction after training: f(5) = {model(X_test).item():.3f}')


# -----------------------------
# 07 Linear Regression
# -----------------------------

# 1) Design model (input, output, size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#     - Step 1：Forward pass
#     - Step 2：Backward pass: Gradient
#     - Step 3：update weight

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
 # data 樣式是numpy
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)
 # 此時，X_numpy.shape = (100, 1); y_numpy.shape = (100,)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # 用意是 轉成1 column 陣列
 # 現在，X.shape = torch.Size([100, 1]); y.shape = torch.Size([100, 1])

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epoch = 10000
for epoch in range(num_epoch):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # backward pass
    loss.backward()
    
    # update # zero gradients
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 500 ==0:
        # [w, b] = model.parameters() # unpack the parameters
        print (f'epoch {epoch+1}: loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy() # 轉回numpy # detach:分離，大概表示分離成一個新的tensor vector
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()


# -----------------------------
# 08 Logistic Regression
# -----------------------------
 """hard to understand""" # 1) & 3)


# 1) Design model (input, output, size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#     - Step 1：Forward pass
#     - Step 2：Backward pass: Gradient
#     - Step 3：update weight

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler # to scale our features
from sklearn.model_selection import train_test_split # separate of training and testing data
import matplotlib.pyplot as plt


# 0) prepare data
 # data 樣式是numpy
bc = datasets.load_breast_cancer() # binary classification problem, predict concept based on features
X,y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
 # 從 datasets 中抽 20% 的 data 當 test data

# scale the features
sc = StandardScaler() # a function always recommended, to have zero mean and unit variance
                      # to deal with logistic regression
X_train = sc.fit_transform(X_train) # sc需要先fit, 才能做transform
X_test = sc.transform(X_test) # 已經fit過了
 
X_train = torch.from_numpy(X_train.astype(np.float32)) # 轉成torch
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1) # reshape 成為 2D 1 column
y_test = y_test.view(y_test.shape[0],1)


# 1) model
 # f = wx + b, a linear combination 
 # in logistic regression case, applied a sigmoid function at the end (S型) 

class LogisticRegression(nn.Module): # 創一個"類別"去"繼承"內建的nn.Module類別
    
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # output feature =1, 1 value (1 class label) at the end
        
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)


# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # Binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epoch = 10000
for epoch in range(num_epoch):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # backward pass
    loss.backward()
    
    # update # zero gradients
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 500 ==0:
        print (f'epoch {epoch+1}: loss = {loss.item():.4f}')

with torch.no_grad(): # evaluate the model
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() # classify, 四捨五入 to become a binary classification.
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) # .eq(y_test) 看y_predicted_cls等不等於y_test
    print(f'accuracy = {acc.item():.4f}')
 

# plot # 影片沒有這一部份
predicted_cls = model(X_test).detach().round().numpy() # 
X_test_numpy = X_test.numpy()

plt.plot(X_train, y_train, 'ro')
plt.plot(X_test_numpy, predicted_cls, 'k.')
plt.show()



# -----------------------------
# 09 Dataset and Dataloader
# -----------------------------
""""
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in 1 forward & backward pass

number of iteration = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch

"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset): # 創一個"類別"去"繼承"內建的Dataset類別
    
    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.txt', delimiter=",", dtype=np.float32, skiprows=1) # delimiter=",": 以逗點分隔(csv格式)；skiprows=1: 略過第一行，因為是header。
        self.x = torch.from_numpy(xy[:, 1:]) # dataset.x.shape = torch.Size([178, 13]) # features
        self.y = torch.from_numpy(xy[:, [0]]) # n_sample, 1 # dataset.y.shape = torch.Size([178, 1]) # labels
        self.n_samples = xy.shape[0] # dataset.n_samples = 178
        
    def __getitem__(self, index): # 這裡表示: 呼叫 dataset[0] 時, 要回復(x[index],y[index])
        # dataset[0]
        return self.x[index], self.y[index]
        # 如果沒寫這行，呼叫 dataset[0] 時, 會得到 NotImplementedError
        
    def __len__(self): # 這裡表示: 呼叫 len(dataset) 時, 要回復n_samples
        # len(dataset)
        return self.n_samples
        # 如果沒寫這行，呼叫 len(dataset) 時, 會得到 TypeError: object of type 'WineDataset' has no len()

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True) # separate dataset into batches

# datatiter = iter(dataloader)  # to make dataloader iterable
# data = datatiter.next()       # 拿一個batch出來
# features, labels=data
# print(features, labels)



# Dummy training loop
num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4) # math.ceil() 無條件進位
print(total_samples, n_iterations)

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Run your training process
        if (i+1) % 5 ==0:
            print(f'Epoch:{epoch+1}/{num_epoch}, Step{i+1}/{n_iterations}| Inputs {inputs.shape}| Labels {labels.shape}')

# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

dataloader = DataLoader(dataset=train_dataset,
                          batch_size=3,
                          shuffle=True)

# look at one random sample
dataiter= iter(dataloader)
data = dataiter.next()
inputs, targets = data
print(inputs.shape, targets.shape)


# -----------------------------
# 10 Dataset Transform
# -----------------------------
'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset): # 創一個"類別"去"繼承"內建的Dataset類別
    
    def __init__(self, transform=None): # transform 是指torch / numpy, defult = None = numpy
        # data loading
        xy = np.loadtxt('./wine.txt', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        
        self.x_data = xy[:, 1:] # 之前是把 torch.from_numpy()寫在這裡
        self.y_data = xy[:, [0]] # 之前是把 torch.from_numpy()寫在這裡
        
        self.transform = transform # here to define the type 
        
    def __getitem__(self, index):
        # dataset[0]
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform: # if not none
            sample = self.transform(sample) # here to define the "transform" function
            # 現在,寫在這裡，
        return sample
        
    def __len__(self):
        # len(dataset)
        return self.n_samples

# Custom Transforms
# implement __call__(self, sample)
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
        # and also here to define the type transform function

class MulTransform: 
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

print('Without Transform')
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)


# -----------------------------
# 11 Softmax and CrossEntropy
# -----------------------------
import torch
import torch.nn as nn
import numpy as np

"""
#        -> 2.0              -> 0.65  
# Linear -> 1.0  -> Softmax  -> 0.25   -> CrossEntropy(y, y_hat)
#        -> 0.1              -> 0.1                   
#
#     scores(logits)      probabilities
#                           sum = 1.0
#

# Softmax applies the exponential function to each element, and normalizes
# by dividing by the sum of all these exponentials
# -> squashes the output to be between 0 and 1 = probability
# sum of all probabilities is 1

"""

# Softmax = e^x / sum(e^x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # 要給維度
print('softmax numpy:', outputs)

# Cross Entropy Loss = - (1/N) Y * log(Y_hat)
def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss

 # y must be one hot encoded
  # if class 0: [1 0 0]
  # if class 1: [0 1 0]
  # if class 2: [0 0 1]
Y = np.array([1, 0, 0])

 # y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss 1 numpy:{l1:.4f}')
print(f'Loss 2 numpy:{l2:.4f}')


# with pytorch
"""
nn.CrossEntropyLoss applies
= nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)

--> No softmax in last layer

Y has class labels, not One-Hot
Y_pred has raw scores (logits), no Softmax!

"""

loss = nn.CrossEntropyLoss()

# 3 samples
Y = torch.tensor([2, 0, 1]) # 2 means index=2 的位置數值大(機率大)

# nsamples * nclasses = 3*3
Y_pred_good = torch.tensor([[1.1, 0.0, 2.1], [2.0, 0.0, 1.1], [0.1, 2.2, 1.0]])
# 所以 Y_pred_good 的 [1.1, 0.0, 2.1] 的 index=2 的位置數值大(機率大)
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [1.1, 0.0, 2.1]])

l1 = loss(Y_pred_good, Y) # 先放prediction, 再放Y
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

# to get actual prediction
_, predictions1 = torch.max(Y_pred_good,1) # 在第一維度找最大值，在一個陣列自己比。(cf 在第零維度找最大值，每個陣列第i的值互比)
_, predictions2 = torch.max(Y_pred_bad,1)
print(predictions1)
print(predictions2)

"""" important """

""" Binary classification: Is it a dog"""
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        """ sigmoid at the end """
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() """"""

""" Multiclass problem: What Animal """
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        """ no softmax at the end """
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() """(applies Softmax) """


# -----------------------------
# 12 Activation Functions
# -----------------------------

# to decide whether the neuron should activate or not
 # Step function _|-    --> not used in practice
 # Sigmoid s            --> typically in the last layer of a binary classification problem
 # TanH s_變化版        --> Hidden layer
 # ReLu _/              --> If don't know what to use, then ues it for hidden layer
 # Leaky ReLu _/ 變化版 --> tried to solve the vanishing gradient problem
 # Softmax              --> good in last layer in multi class classification problems
 
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

# sofmax
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

# sigmoid 
output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

#tanh
output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)

# relu
output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)

# leaky relu
output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)


#nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
#torch.relu on the other side is just the functional API call to the relu function,
#so that you can add it e.g. in your forward method yourself.

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
    

# -----------------------------
# 13 Feed Forward Net
# -----------------------------
# MNIST dataset
# DataLoader, Transformation
# Multilayer Neural Net (input, hiden, output), activation function
# Loss and Optimizer
# Training Loop (Batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device configuration # if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # 28*28 , turn into 1D
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
# check for the parameter: https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False,
                                          transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader) # 使test_loader 套入可迭代物件: https://ithelp.ithome.com.tw/articles/10196096
example_data, example_targets = examples.next() # 啟動前一步的迭代物件，叫出第一個 batch 的資料

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) # first layer
        self.relu = nn.ReLU() # activaation function on hiden layer
        self.l2 = nn.Linear(hidden_size, num_classes) # last layer
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end, cause its multiclass
        # and we will use nn.CrossEntropyLoss() later
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader) # = 600 = how many batches
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # enumerate() will give the i, (images, labels)tuple
        # origin shape:[100, 1, 28, 28]
        # resized:[100, 784]
        images = images.reshape(-1, 28*28).to(device) # put to GPU if avaliable
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch[{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')
            

# Test the model
# In test phase, we don't need to compute gradients (for memory efficency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # torch.max() return(value, index), and we just need the index
        _, predicted = torch.max(outputs.data, 1) # so there is the underscore
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test image: {acc} %')


# -----------------------------
# 14 CNN convolution neural network
# -----------------------------

# CIFAR10 包括貓、狗、車、船等10種圖片
# convolution neural network 步驟:
    # convolution --> ReLU --> (Max) Pooled --> convolution...
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 20
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]) # https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=transforms%20normalize#torchvision.transforms.Normalize

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
# 每筆資料型態是(2, 3, 32, 32)，第一維度的2 包含image data，和label。
train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                           train=True, 
                                           transform=transform, 
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                          train=False,
                                          transform=transform)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy() # 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(train_loader) # 使資料成為可迭代資料(套入可迭代物件)
images, labels = dataiter.next() # 啟動迭代功能
# images 資料維度是 [4, 3, 32, 32] 4圖片*3顏色*32*32資料點

# show images
imshow(torchvision.utils.make_grid(images))

# my test to show the difference
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i][0])
plt.show()
# 無法累加圖層成彩圖


# implement convolution net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 42, 3) # input channel size: 3 color channel, output channel size: 42, kernel size: 3
        self.conv2 = nn.Conv2d(42, 84, 3) # the input channel size = the last output channel size = 42
        self.conv3 = nn.Conv2d(84, 168, 3)
        self.pool = nn.MaxPool2d(2,2) # kernel size: 2 --> 2*2 的frame, stride: 2 --> 一步走2格，不重複 --> MaxPool
        self.fc1 = nn.Linear(168*2*2, 168) # 後者 168 自己設的output cell 數
        self.fc2 = nn.Linear(168, 84) # 168 上一步設的output cell 數，84 自己設的output cell 數
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, 10) 
        # self.dropout1 = nn.Dropout(p=0.2, inplace=False) # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        
    def forward(self, x):
        # --> 4, 3, 32, 32 ; (W-F+2P)/S + 1, W: 32*32 input, F: 5*5 fliter(kernel), P: padding = 0, S: stride = 1
        x = self.pool(F.relu(self.conv1(x))) # --> 4, 42, 30, 30 --> 4, 42, 15, 15
        # x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x))) # --> 4, 84, 13, 13 --> 4, 84, 6, 6
        # x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x))) # --> 4, 168, 4, 4 --> 4, 168, 2, 2
        # x = self.dropout1(x)
        x = x.view(-1, 168*2*2)               # --> 4, 672
        x = F.relu(self.fc1(x))              # --> 4, 168
        x = F.relu(self.fc2(x))              # --> 4, 84
        x = F.relu(self.fc3(x))              # --> 4, 42
        x = self.fc4(x)                      # --> 4, 10
        return x
    

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader) # =12500, 因為50000張 / 4張 per batch, 另10000
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # enumerate() 輸出的可迭代對象的數值以及他們的編號
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2000 ==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

# calculate the accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)] # a list of 10 zeros
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (output value, label index)
        _,predicted = torch.max(outputs, 1)
        # dim=1, 四個矩陣分別取最大值及其位置 (位置index就會=label); 
        # dim=0, 四個矩陣同個位置互相比較並選出最大值, index紀錄最大值來自哪個矩陣。
        n_samples += labels.size(0) # labels.size(0)=4, 因為四張圖片per batch
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

# -----------------------------
# 15 Transfer Learning
# -----------------------------
# ImageFolder
# Scheduler --> to change learning rate
# Transfer Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import os

import time
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    
    }

data_dir = ''


# Oh... no data provided... pause


# -----------------------------
# 16 Tensor Board
# -----------------------------
# from 13 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist2')
###################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

############## TENSORBOARD ########################
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
#writer.close()
#sys.exit()
###################################################

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

############## TENSORBOARD ########################
writer.add_graph(model, example_data.reshape(-1, 28*28))
#writer.close()
#sys.exit()
###################################################

# Train the model
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            ############## TENSORBOARD ########################
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            running_accuracy = running_correct / predicted.size(0)
            writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0
            ###################################################

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
class_labels = []
class_preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        values, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_probs_batch)
        class_labels.append(predicted)

    # 10000, 10, and 10000, 1
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    ############## TENSORBOARD ########################
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    ###################################################































