# -*- coding: utf-8 -*-

"""
使用tensorflow进行线性回归(Linear regression)
输入X=(x1,x2....,xn)
输出Y=(y1,y2...,ym)
映射函数关系：
            Y=WX+b
W为m*n大小的权重矩阵
B为m*1的偏置向量

机器学习的目标的通过已有的输入样本(X,Y)来最小化目标化函数：
target function:loss=0.5*sum((WX+B-Y)^2)
找到能够使loss最小的W和B
let's define the model
"""

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
import tensorflow as tf


# 生成traing dataset 并plot
def inputs():
    X = [x + random.random() for x in range(100)]
    # 生成的模型满足:线性主要部分+残差
    Y = [[0.638 * x + 0.55 + np.random.normal(0, 0.1)] for x in X]
    X = [[x] for x in X]
    return X, Y


X_in, Y_in = inputs()
plt.plot(X_in, Y_in, 'or') # 根据输入点画出一条直线

# 输入的占位符
x_in = tf.placeholder("float", [None, 1], name="input_x")
y_in = tf.placeholder("float", [None, 1], name="input_y")

# 待学习的参数
W = tf.Variable(0., name="weights")
b = tf.Variable(0., name="bias")


# function:wx+b
def inference(X):
    """
    线性函数
    :param X:
    :return:
    """
    return tf.mul(X, W) + b


# 以均方误差定义损失函数
def loss(X, Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


# 一次训练
global_step = tf.Variable(0)


def train(total_loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)


# 输入X并预测Y值
def evaluate(sess, X):
    return sess.run(inference(X))


total_steps = 500
learning_rate = 0.0000001
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    total_loss = loss(x_in, y_in)

    total_loss_summary = tf.scalar_summary(b'total_loss', total_loss, name="total_loss")
    writer = tf.train.SummaryWriter('./LinearRegression_graph', sess.graph)

    train_step = train(total_loss, learning_rate)
    feed_dict = {x_in: X_in, y_in: Y_in}

    for i in range(total_steps):
        _, step, summary = sess.run([train_step, global_step, total_loss_summary], feed_dict)

        writer.add_summary(summary, global_step=step)

    writer.close()
    print W.eval()
    print b.eval()

    Y_predict = [evaluate(sess, x) for x in X_in]

plot2 = plt.plot(X_in, Y_predict, label='model prediction')
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FFCC')
plt.show()
