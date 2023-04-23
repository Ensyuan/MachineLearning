# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:25:27 2023

@author: Ensyuan
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
 # np.linspace 從 a 到 b 均切 x 點 (含頭尾)
 # np.vstack (vertical stack) 縱向堆疊

# y = 2(x^2)+1
plt.plot(PAINT_POINTS[0], 2*np.power(PAINT_POINTS[0], 2)+1, c='#74BCFF', lw=3, label='upper bound')
# y = 1(x^2)+0
plt.plot(PAINT_POINTS[0], 1*np.power(PAINT_POINTS[0], 2)+0, c='#FF9359', lw=3, label='upper bound')
plt.legend(loc='upper right')
plt.show()

def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis] # [:, np.newaxis] 從(64,) 變成 (64, 1)
    paintings = a*np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


# Generator network
G = nn.Sequential(nn.Linear(N_IDEAS, 128), 
                  nn.ReLU(),
                  nn.Linear(128, ART_COMPONENTS))
                  
# Discriminator network
D = nn.Sequential(nn.Linear(ART_COMPONENTS, 128),
                  nn.ReLU(),
                  nn.Linear(128, 1),
                  nn.Sigmoid())

opt_D = torch.optim.Adam(D.parameters(), lr = LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr = LR_G)


plt.ion() # something about continuous plotting

for step in range(10000):
    artist_paintings = artist_works()
    
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    G_paintings = G(G_ideas)
    
    prob_artist1 = D(G_paintings)
    
    G_loss = torch.mean(torch.log(1.0 - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    prob_artist0 = D(artist_paintings)  # D try to increase this prob??
    prob_artist1 = D(G_paintings.detach()) # .detach()?? # D try to reduce this prob??
    
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1.0 - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()
    
    if step % 50 ==0:
        plt.cla() # ??
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label=' bound')
        plt.text(-.5, 2, 'D accuracy = %.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size':13})
        plt.text(-.5, 1.7, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()    
    









