# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:32:47 2023

@author: Ensyuan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 20
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                         train=True, 
                                         transform=transform, 
                                         download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=transform,
                                        download=False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
# utilized in the end for calculating the class accuracy.


def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))  
    plt.show()
    
dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(images) # axes don't match array 因為有四維:images.shape = torch.Size([4, 3, 32, 32])
imshow(torchvision.utils.make_grid(images))
# torchvision.utils.make_grid(images) 將圖片整併成一張，並加上格線。 torch.Size([3, 36, 138])


# CNN network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 42, 3)
        self.conv2 = nn.Conv2d(42, 84, 3)
        self.conv3 = nn.Conv2d(84, 168, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(168*2*2, 168)
        self.fc2 = nn.Linear(168, 84)
        self.fc3 = nn.Linear(84, 42)
        self.fc4 = nn.Linear(42, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 168*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_step = len(train_loader)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2000 ==0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_step}], Loss: {loss.item():.4f}')
            
print('Finish Training')
PATH = './cnn.pth'
torch.save(model.state_dict, PATH)


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, prediction = torch.max(outputs, 1)
        
        n_samples += labels.size(0)
        n_correct += (prediction == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = prediction[i]
            
            if (label == pred):
                n_class_correct[label] +=1
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print (f'Accuracy of the network: {acc}%')
    
    for i in range(10):
        class_acc = n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of classes {classes[i]} is: {class_acc}')
        

        
    
    
    
        


    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


