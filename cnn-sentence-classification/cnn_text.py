# -*- coding: utf-8 -*-
import tensorflow as tf

class TextCNN(object):
    """
    主要结构：embedding+conv+maxp+softmax
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # 1. 定义输入和输出变量
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # 保留权值比例

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 2. 构建WE层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # 这里是什么意思,词向量变成了什么 TODO？
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 3.针对多组filter构建conv+maxp结构
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # like NHeitht_Width_C style,using like 2d
                filter_shape = [filter_size, embedding_size, 1, num_filters] # 计算一下conv的filter的大小
                # 正态分布初始权值
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # conv
                conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,  # 这个是input
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                # relu
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # maxp-k-pool
                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],  # max-k-pool
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                pooled_outputs.append(pooled)

        # 4. 拼接所有的pool特征+fc全连接层
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)  # 最后一维进行拼接
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 隐层后加入dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 5. softmax的输出层
        with tf.name_scope("output"):
            W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            # 针对分类层进行正则化
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            # drop_w*W+b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # softmax-index layer
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 6-1 定义交叉的损失 cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            # 总的损失函数的计算
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 6-2.定义准确率
        with tf.name_scope("accuracy"):
            # 总的精度的计算
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
