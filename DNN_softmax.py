import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Чтение датасета
mnist = input_data.read_data_sets(".", reshape=True, one_hot=True)
#print(mnist.train.images.shape)

# Остовные параметры для обучения
logs_path = 'log_softmax'
batch_size = 100
learning_rate = 0.5
training_epochs = 16

# Вектор независимых переменных (пиксели)
X = tf.placeholder(
	tf.float32,
	[None, 784],
	name="imput"
)

# Вектор зависимых переменных (класс 0, 1, 2, ..., 9
Y_ = tf.placeholder(tf.float32, [None, 10])

# Матрица весов
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# На выходе из первого (и единственного) слоя нейронов
evidence = tf.add(
	tf.matmul(X, W),
	b
)
Y = tf.nn.softmax(evidence, name="output")

# Оценка ошибки, которую нужно уменьшить в процессе обучения
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(
		logits=evidence,
		labels=Y_
	)
)

# Обучающий алгоритм
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Определение корректного предсказания
correct_prediction = tf.equal(
	tf.argmax(Y, 1),
	tf.argmax(Y_, 1)
)

# Вычисление точности / accuracy
accuracy = tf.reduce_mean(
	tf.cast(
		correct_prediction,
		tf.float32
	)
)

# Статистика
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

# Определение сессии
sess = tf.Session()

# Инициализация переменных
sess.run(tf.global_variables_initializer())

# Настройка записи статистики в логи
writer = tf.summary.FileWriter(
	logdir=logs_path,
	graph=tf.get_default_graph()
)

# ОБУЧЕНИЕ ::
for epoch in range(training_epochs):
	batch_count = int(mnist.train.num_examples / batch_size)
	print("\nEpoch {}".format(epoch+1))
	for i in range(batch_count):
		batch_x, batch_y = mnist.train.next_batch(batch_size)

		_, summary = sess.run(
			[train_step, summary_op],
			feed_dict={
				X: batch_x,
				Y_: batch_y
			}
		)
		writer.add_summary(summary, epoch * batch_count + 1)

	print(">> Accuracy {}".format(
		sess.run(
			accuracy,
			feed_dict={
				X: mnist.test.images,
				Y_: mnist.test.labels
			}
		)
	))

	# Предсказание класса для одной картинки из датасета
	print(
		np.argmax(
			sess.run(
				Y,
				feed_dict={X: mnist.test.images[6:7]}

			)[0]
		)
	)
	print(mnist.test.labels[6:7])


