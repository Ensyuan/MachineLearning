# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 06:55:34 2023

@author: Ensyuan
"""

## file:///D:/%E8%A8%88%E7%AE%97%E7%A5%9E%E7%B6%93%E7%A7%91%E5%AD%B8109A/final_%E9%BB%83%E6%81%A9%E7%92%BF.pdf
## file:///D:/計算神經科學109A/final_黃恩璿.pdf
import numpy as np
import matplotlib.pyplot as plt
import struct
import gzip

# read training data
with gzip.open('train-labels-idx1-ubyte.gz','rb') as f:
    buf = f.read()
d1 = np.frombuffer(buf[8:], dtype = np.uint8).newbyteorder(">") ## 意思是？

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    buf = f.read()
magic, size, nrows, ncols = struct.unpack('>IIII', buf[:16]) ## '>IIII'意思是？
dat = np.frombuffer(buf[16:], dtype=np.uint8).newbyteorder(">") ## 意思是？
dat = dat.reshape((size, nrows, ncols))

print(d1[:5])
f, aa = plt.subplots(1, 5, figsize=(10,2))
for i, a in zip(dat[:5], aa):
    a.imshow(i)
plt.show()


# binarize and pick up only 0 and 1
imtry0 = (dat[d1==0])
imtry1 = (dat[d1==0]>128) ## 是boolin值 (T/F)
im0 = (dat[d1==0]>128)*2-1 ## 
im1 = (dat[d1==1]>128)*2-1

f, aa = plt.subplots(1, 5, figsize=(10, 2))
for i, a in zip(im0[:5], aa):
    a.imshow(i)
plt.show()

f, aa = plt.subplots(1, 5, figsize=(10, 2))
for i, a in zip(im1[:5], aa):
    a.imshow(i)
plt.show()


#-------------------------------
#  Problem 1
#  Calculate weight for the perception
#-------------------------------

w2 = (np.sum(im1, axis=0)-np.sum(im0, axis=0)) # 製作權重
plt.imshow(w2)


#-------------------------------
#  Problem 2
#  Load the testing data files 
#-------------------------------

with gzip.open('t10k-labels-idx1-ubyte.gz','rb') as f:
    buf = f.read()
d2 = np.frombuffer(buf[8:], dtype = np.uint8).newbyteorder(">") ## 意思是？

with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    buf = f.read()
magic, size, nrows, ncols = struct.unpack('>IIII', buf[:16]) ## '>IIII'意思是？
dat2 = np.frombuffer(buf[16:], dtype=np.uint8).newbyteorder(">") ## 意思是？
dat2 = dat2.reshape((size, nrows, ncols))

# binarize and pick up only 0 and 1
test_im0 = (dat2[d2==0]>128)*2-1 
test_im1 = (dat2[d2==1]>128)*2-1

f, aa = plt.subplots(1, 5, figsize=(10, 2))
for i, a in zip(test_im0[:5], aa):
    a.imshow(i)
plt.show()

f, aa = plt.subplots(1, 5, figsize=(10, 2))
for i, a in zip(test_im1[:5], aa):
    a.imshow(i)
plt.show()


# calculate the fraction of correct perception of the percception

# 稍微懂了
n = 1000
ans = np.random.binomial(1, 0.5, size=n) # 產生隨機0101
test = np.take([test_im0, test_im1], ans, axis=0) # 按著 ans的0101取test_im0或 test_im1

op_list = []
for i in range(0,n,1):
    output = (np.mean(test[i]*w2, axis=(0, 1, 2))>0).astype(int)==ans[i]
    op_list.append(output)
print('the fraction of correct prediction:', np.sum(op_list)/n)


op_list2 = []
for i in range(0,n,1):
    output2 = (np.mean(test[i]*w2, axis=(0, 1, 2))>200).astype(int)==ans[i] # >200 與 >0 是自己要找出的threshold
    op_list2.append(output2)
print('the fraction of correct prediction:', np.sum(op_list2)/n)



#-------------------------------
#  bonus
#-------------------------------


















