import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

train_xdata = mnist.train.images
test_xdata = mnist.test.images

train_labels = mnist.train.labels
test_labels = mnist.test.labels

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = 10
num_channels = 1
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

x_input_shape = (
    batch_size,
    image_width,
    image_height,
    num_channels
)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_input = tf.placeholder(tf.float32, shape=(batch_size))

eval_input_shape = (
    evaluation_size,
    image_width,
    image_height,
    num_channels
)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.float32, shape=evaluation_size)

conv1_weight = tf.Variable(
    tf.truncated_normal(
        [4, 4, num_channels, conv1_features],
        stddev=0.1,
        dtype=tf.float32
    )
)
conv1_bias = tf.Variable(
    tf.zeros(
        [conv1_features],
        dtype=tf.float32
    )
)

conv2_weight = tf.Variable(
    tf.truncated_normal(
        [4, 4, conv1_features, conv2_features],
        stddev=0.1,
        dtype=tf.float32
    )
)
conv2_bias = tf.Variable(
    tf.zeros(
        [conv2_features],
        dtype=tf.float32
    )
)


resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_height * resulting_width * conv2_features

full1_weight = tf.Variable(
    tf.truncated_normal(
        [full1_input_size, fully_connected_size1],
        stddev=0.1,
        dtype=tf.float32
    )
)
full1_bias = tf.Variable(
    tf.truncated_normal(
        [fully_connected_size1],
        stddev=0.1,
        dtype=tf.float32
    )
)

full2_weight = tf.Variable(
    tf.truncated_normal(
        [fully_connected_size1, target_size],
        stddev=0.1,
        dtype=tf.float32
    )
)

full2_bias = tf.Variable(
    tf.truncated_normal(
        [target_size],
        stddev=0.1,
        dtype=tf.float32
    )
)

def my_conv_net(input_data):
    conv1 = tf.nn.conv2d(
        input_data,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )
    