import tensorflow as tf
from sklearn.datasets import fetch_mldata
import numpy as np 
from sklearn.model_selection import train_test_split

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)  
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train = y_train.astype(np.int32) 
y_test = y_test.astype(np.int32)

height = 28 
width = 28 
channels = 1 
n_inputs = height * width 

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 
X_reshape = tf.reshape(X, shape = [-1, height, width, channels]) 
y = tf.placeholder(tf.int32, shape = [None]) 

conv1 = tf.layers.conv2d(inputs = X_reshape, filters = 32, kernel_size = [5,5],
						padding="same", activation = tf.nn.relu) 
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides = 2) 
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding="same",
						activation=tf.nn.relu) 
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2) 
pool2_flat = tf.reshape(pool2, [-1, 7*7*64]) 
fcl1 = tf.layers.dense(inputs=pool2_flat, units = 1024) 

logits = tf.layers.dense(inputs=fcl1, units = 10) 

with tf.name_scope("train"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)  
	loss = tf.reduce_mean(xentropy) 
	optimizer = tf.train.AdamOptimizer() 
	training_op = optimizer.minimize(loss) 

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

n_epoches = 1
batch_size = 100 
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch

init = tf.global_variables_initializer()

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epoches):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) 
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) 
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test}) 
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test) 



