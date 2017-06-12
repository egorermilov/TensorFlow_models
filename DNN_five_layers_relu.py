import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#import math

# Остовные параметры для обучения
logs_path = "log_five_layers"
batch_size = 100
learning_rate = 0.003
training_epochs = 10

# Чтение датасета
mnist = input_data.read_data_sets(".", reshape=True, one_hot=True)

# Вектор независимых переменных (пиксели)
X = tf.placeholder(
    tf.float32,
    [None, 784],
    name="imput"
)

# Вектор зависимых переменных (класс 0, 1, 2, ..., 9
Y_ = tf.placeholder(tf.float32, [None, 10])

# Размеры скрытых слоёв нейронной сети
L = 200
M = 100
N = 60
O = 30

# Архитектура нейронной сети ::
W1 = tf.Variable(
    tf.truncated_normal([784, L], stddev=0.1)
)
B1 = tf.Variable(tf.zeros([L]))
Y1 = tf.nn.relu(
    tf.add(
        tf.matmul(X, W1),
        B1
    )
)

W2 = tf.Variable(
    tf.truncated_normal([L, M], stddev=0.1)
)
B2 = tf.Variable(tf.zeros([M]))
Y2 = tf.nn.relu(
    tf.add(
        tf.matmul(Y1, W2),
        B2
    )
)

W3 = tf.Variable(
    tf.truncated_normal([M, N], stddev=0.1)
)
B3 = tf.Variable(tf.zeros([N]))
Y3 = tf.nn.relu(
    tf.add(
        tf.matmul(Y2, W3),
        B3
    )
)

W4 = tf.Variable(
    tf.truncated_normal([N, O], stddev=0.1)
)
B4 = tf.Variable(tf.zeros([O]))
Y4 = tf.nn.relu(
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

# Оценка ошибки, которую нужно уменьшить в процессе обучения
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
) * 100

# Обучающий алгоритм
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Определение корректного предсказания
correct_prediction = tf.equal(
    tf.argmax(Y, 1),
    tf.argmax(Y_, 1)
)

# Вычисление точности / accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Статистика
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

# Определение сессии
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Настройка записи статистики в логи
writer = tf.summary.FileWriter(
    logdir=logs_path,
    graph=tf.get_default_graph()
)

# ОБУЧЕНИЕ ::
for epoch in range(training_epochs):
    batch_count = int(mnist.train.num_examples / batch_size)
    print("\nEpoch {}".format(epoch + 1))
    for i in range(batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, summary = sess.run(
            [train_step, summary_op],
            feed_dict={
                X: batch_x,
                Y_: batch_y
            }
        )
        writer.add_summary(summary, epoch * batch_count + i)

    print(">> Accuracy {}".format(
        sess.run(
            accuracy,
            feed_dict={
                X: mnist.test.images,
                Y_: mnist.test.labels
            }
        )
    ))

