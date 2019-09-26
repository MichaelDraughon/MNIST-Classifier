# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:21:39 2019

@author: Michael
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()

fig, axes = plt.subplots(1, 4, figsize = (7, 3))

for img, label, ax in zip(xTrain[:4], yTrain[:4], axes):
    ax.set_title(label)
    ax.imshow(img)
    ax.axis('off')
plt.show()

#preprocessing
xTrain = xTrain.reshape(60000, 784) / 255
xTest = xTest.reshape(10000, 784) / 255

with tf.Session() as sess:
    yTrain = sess.run(tf.one_hot(yTrain, 10))
    yTest = sess.run(tf.one_hot(yTest, 10))
    
#hyper parameters
learningRate = 0.001
epochs = 100
batchSize = 100
batches = int(xTrain.shape[0] / batchSize)

#data
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#variables that will change as the AI learns
W = tf.Variable(0.1 * np.random.randn(784, 10).astype(np.float32))
B = tf.Variable(0.1 * np.random.randn(10).astype(np.float32))

#setting up cost function and training algorithm
pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

#training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * epoch
            x = xTrain[offset: offset + batchSize]
            y = yTrain[offset: offset + batchSize]
            sess.run(optimizer, feed_dict = {X: x, Y: y})
            c = sess.run(cost, feed_dict = {X: x, Y: y})
        if not epoch % 1:
            print(f'epoch:{epoch} cost={c:.4f}')
    correctPred = tf.equal(tf.math.argmax(pred, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    acc = accuracy.eval({X: xTest, Y: yTest})
    print(str(acc * 100) + "% accuracy")
    