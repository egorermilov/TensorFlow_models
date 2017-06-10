import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", reshape=True, one_hot=True)
#print(mnist.train.images.shape)

logs_path = 'log_softmax'
batch_size = 100
learning_rate = 0.5
training_epochs = 16

X = tf.placeholder(
	tf.float32,
	[None, 784],
	name="imput"
)
Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

evidence = tf.add(
	tf.matmul(X, W),
	b
)
Y = tf.nn.softmax(evidence, name="output")

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(
		logits=evidence,
		labels=Y_
	)
)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(
	tf.argmax(Y, 1),
	tf.argmax(Y_, 1)
)

accuracy = tf.reduce_mean(
	tf.cast(
		correct_prediction,
		tf.float32
	)
)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(
	logdir=logs_path,
	graph=tf.get_default_graph()
)

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
	sess.run(
		Y,
		feed_dict={X: mnist.test.images[6:6]}

	)


