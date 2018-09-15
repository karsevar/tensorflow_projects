import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_mldata 
from datetime import datetime
import matplotlib.pyplot as plt 

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]

print(28 * 28)

plt.imshow(X_mnist[1].reshape(28, 28))
plt.show()  

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=20000, random_state = 42, shuffle = True)
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train = y_train.astype(np.int32) 
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:] 
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28 * 28 
n_hidden1 = 100
n_hidden2 = 100 
n_outputs = 10 
learning_rate = 0.01 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.elu) 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.elu)
	logits = tf.layers.dense(hidden2, n_outputs, name = "logits")  

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate) 
	training_op = optimizer.minimize(loss)

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

n_epochs = 30 
batch_size = 50
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs" 
logdir = "{}/run-{}/".format(root_logdir, now) 
final_model_path = "./my_deep_mnist_model"

accuracy_summary = tf.summary.scalar("accuracy", accuracy) 
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)

			if batch_index % 10 == 0:
				summary_str = accuracy_summary.eval(feed_dict={X: X_batch, y: y_batch})
				step = epoch * n_batches + batch_index 
				file_writer.add_summary(summary_str, step) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid)

	saver.save(sess, final_model_path)  

file_writer.close() 
tf.reset_default_graph()