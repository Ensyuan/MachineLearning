# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:39:38 2022

@author: Ensyuan
"""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
print(hello)

# =================================================
# Keras-MNIST-數字辨識
# =================================================

import tensorflow as tf

#將MNIST 手寫數字資料讀進來
mnist = tf.keras.datasets.mnist

# mnist 的load_data()會回傳已經先分割好的training data 和 testing data
# 並且將每個 pixel 的值從 Int 轉成 floating point 同時做normalize
# (這是很常見的preprocessing)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # 將數字從 0~255 之間轉成 0~1 之間

print(len(x_train))
print(x_train[0].shape)
print(x_train[0])

#--------- 印出第一張 ----------
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()

#-----------Model 1-------------

# 開始搭建model
# 利用 "Sequential" 把每層 layer 疊起來
# input 大小為 28 x 28
# 最後的 Dense(10) 且 activation 用 softmax
# 代表最後 output 為 10 個 class （0~9）的機率


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(128, activation='relu'), 
                                    tf.keras.layers.Dropout(0.2), 
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])

print(model.summary())

# model每層定義好後需要經過compile(編輯)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#-----------Model 2-------------

# 開始搭建model
# 利用 "Sequential" 把每層 layer 疊起來
# input 大小為 28 x 28
# 最後的 Dense(10) 且 activation 用 softmax
# 代表最後 output 為 10 個 class （0~9）的機率


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(128, activation='relu'), 
                                    tf.keras.layers.Dropout(0.8), 
                                    tf.keras.layers.Dense(64, activation='relu'), 
                                    tf.keras.layers.Dropout(0.8),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])

# model每層定義好後需要經過compile(編輯)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 比較:
    # tf.keras.layers.Dense 多加一層沒有比較精準
    # tf.keras.layers.Dropout 數字上升，精準度下降

#-------------------------------

# 將搭好的 model 去 fit 我們的 training data
# 並evalutate 在 testing data 上

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2) # verbose ??

# 會自己print每個epoch的accuracy 很棒!



# =================================================
# Keras–MNIST–自訂模型
# =================================================

# 重點
# 1. GradientTape
# 2. training 基本流程
# 3. tf.function (decorator)
# 4. training data shape

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # ??

x_train.shape


# Add a channels dimension，多增加第4個 dimension:"channel"
# for what
## --> 每個點的數值都自成一個維度了!!
## 也許是因為data需要兩個維度以上
## pytorch data也是 @ lesson 6
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

x_train.shape
print(x_train[0])


# ----------- tf.data API 來建立 input pipeline -------------
#  * tf.data

# 1."from_tensor_slices" 建立一個 Dataset Object
#   在使用"from_tensor_slices"參數（x_train, y_train）
#   是包含很多筆類型一樣的data,像這個例子有 60000筆data。
#   不要跟"from_tensor"搞混了!

# 2.將training data做shuffle打亂 和 將batch size設定為 32

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32)


# ------------- 使用 Keras 自訂一個 Model --------------
#  * tf.keras.Model
#  * tf.keras.layers

# 1.建立一個 Class並繼承 tf.Keras.Model

# 2.__init__裡面我們可以自己設計每一層 Layer長什麼樣子，
#   是哪種 Layer(Convolution, dense...)

# 3.call定義了這個 Model的 forward pass

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your layers here.
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        
    def call(self, x):
        # Define your forward pass here,
        # using layers you previously defined (in __init__).
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()


# ------------- 選擇 loss, optimizer和 metrics --------------
#  * tf.keras.losses
#  * tf.keras.optimizers
#  * tf.keras.metrics

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# ------------- train the model (定義) --------------
#  * tf.function
#  * tf.GradientTape

@tf.function
def train_step(images, labels):
    
    #
    with tf.GradientTape() as tape:
        # forward pass to get predictions
        predictions = model(images)
        # compute loss with the ground and our predictions
        loss = loss_object(labels, predictions)
    
    # compute gradient
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # back propagation
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    

# ------------- test the model (定義) --------------   
#  * tf.function

@tf.function
def test_step(images, labels):
    
    # forward pass to get predictions
    predictions = model(images)
    # compute loss with the ground and our predictions
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
#---------------- 真正開始 train model ------------------

# 定義 EPOCH 數 (整個dataset 拿來train過一次稱一個epoch)

EPOCH = 5

for epoch in range(EPOCH):
    for images, labels in train_dataset:
        train_step(images, labels)
        
    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)
        
    template = 'Epoch {}, Loss {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}\n'
    print(template.format(epoch+1, 
                          train_loss.result(), 
                          train_accuracy.result()*100, 
                          test_loss.result(), 
                          test_accuracy.result()*100))
    
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    



# =================================================
# SavedModel 儲存模型
# =================================================

# tf.saved_model.save()
# tf.saved_model.load()

tf.saved_model.save(model, 'Grandma')

new_model = tf.saved_model.load('Grandma')

# 定義一個evaluate model 的 function
def evaluate(model):
    for images, labels in test_dataset:
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        
    print('test loss: {}'.format(test_loss.result()))
    print('test accuracy: {}'.format(test_accuracy.result()))
        
    test_loss.reset_states()
    test_accuracy.reset_states()
    
evaluate(model)

evaluate(new_model)
































