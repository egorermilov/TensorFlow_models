import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 100
img_size = 28
num_classes = 10

X = tf.placeholder(tf.float32, [None, img_size, img_size, 1])

Y = tf.placeholder(tf.float32, [None, num_classes])

mnist = input_data.read_data_sets(".", reshape=False, one_hot=True)

trX = mnist.train.images
trY = mnist.train.labels
teX = mnist.test.images
teY = mnist.test.labels

trX = trX.reshape(-1, img_size, img_size, 1)
teX = teX.reshape(-1, img_size, img_size, 1)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.01))

w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_o = init_weights([625, num_classes])

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

def my_model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    conv1 = tf.nn.dropout(
        tf.nn.max_pool(
            tf.nn.relu(
                tf.nn.conv2d(
                    X,
                    w,
                    strides=[1, 1, 1, 1],
                    padding="SAME"
                )
            ),
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        ),
        p_keep_conv
    )

    conv2 = tf.nn.dropout(
        tf.nn.max_pool(
            tf.nn.relu(
                tf.nn.conv2d(
                    conv1,
                    w2,
                    strides=[1, 1, 1, 1],
                    padding="SAME"
                )
            ),
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        ),
        p_keep_conv
    )
    conv3 = tf.nn.max_pool(
        tf.nn.relu(
            tf.nn.conv2d(
                conv2,
                w3,
                strides=[1, 1, 1, 1],
                padding="SAME"
            )
        ),
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    FC_layer = tf.nn.dropout(
        tf.reshape(
            conv3,
            [-1, w4.get_shape().as_list()[0]]
        ),
        p_keep_hidden
    )

    output_layer = tf.nn.dropout(
        tf.nn.relu(tf.matmul(FC_layer, w4)),
        p_keep_hidden
    )

    result = tf.matmul(output_layer, w_o)

    return result

py_x = my_model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
)

optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(py_x, 1)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for epoch in range(15):
    print("\nEpoch {} ...".format(epoch+1))
    training_batch = zip(range(0, len(trX), batch_size),
                         range(batch_size,len(trX)+1, batch_size))
    for start, end in training_batch:
        sess.run(
            optimizer,
            feed_dict={
                X: trX[start:end],
                Y: trY[start: end],
                p_keep_conv: 0.8,
                p_keep_hidden: 0.5
            }
        )
    test_indices = np.arange(len(teX))
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]

    print(
        ">> accuracy = {}".format(
            np.mean(
                np.argmax(teY[test_indices], axis=1) ==
                sess.run(
                    predict_op,
                    feed_dict={
                        X: teX[test_indices],
                        Y: teY[test_indices],
                        p_keep_conv: 1.0,
                        p_keep_hidden: 1.0
                    }
                )
            )
        )

    )




