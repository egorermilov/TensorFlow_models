import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import math

logs_path = "log_five_layers"
batch_size = 100
learning_rate = 0.003
training_epochs = 10

mnist = input_data.read_data_sets(".", reshape=True, one_hot=True)

X = tf.placeholder(
    tf.float32,
    [None, 784],
    name="imput"
)

Y_ = tf.placeholder(tf.float32, [None, 10])

L = 200
M = 100
N = 60
O = 30


W1 = tf.Variable(
    tf.truncated_normal([784, L], stddev=0.1)
)
B1 = tf.Variable(tf.zeros([L]))
Y1 = tf.sigmoid(
    tf.add(
        tf.matmul(X, W1),
        B1
    )
)

W2 = tf.Variable(
    tf.truncated_normal([L, M], stddev=0.1)
)
B2 = tf.Variable(tf.zeros([M]))
Y2 = tf.sigmoid(
    tf.add(
        tf.matmul(Y1, W2),
        B2
    )
)

W3 = tf.Variable(
    tf.truncated_normal([M, N], stddev=0.1)
)
B3 = tf.Variable(tf.zeros([N]))
Y3 = tf.sigmoid(
    tf.add(
        tf.matmul(Y2, W3),
        B3
    )
)

W4 = tf.Variable(
    tf.truncated_normal([N, O], stddev=0.1)
)
B4 = tf.Variable(tf.zeros([O]))
Y4 = tf.sigmoid(
    tf.add(
        tf.matmul(Y3, W4),
        B4
    )
)

W5 = tf.Variable(
    tf.truncated_normal([O, 10], stddev=0.1)
)
B5 = tf.Variable(tf.zeros([10]))
Ylogits = tf.add(
    tf.matmul(Y4, W5),
    B5
)
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
) * 100

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(
    tf.argmax(Y, 1),
    tf.argmax(Y_, 1)
)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))























