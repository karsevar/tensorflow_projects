from datetime import datetime
import tensorflow as tf 
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler 

boston = load_boston() 
m, n = boston.data.shape
print(boston.target.reshape(-1, 1)) 

boston_plus_bias = np.c_[np.ones((m, 1)), boston.data]

scaler = StandardScaler() 
scaler.fit(boston_plus_bias) 
boston_plus_bias_normalized = scaler.transform(boston_plus_bias) 

n_epochs = 10000 
learning_rate = 0.03 

now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

X = tf.placeholder(tf.float32, shape = (None, n + 1), name = "X")  
y = tf.placeholder(tf.float32, shape = (None, 1), name = "y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
gradients = 2/m * tf.matmul(tf.transpose(X), error) 
training_op = tf.assign(theta, theta - learning_rate * gradients)
mse_summary = tf.summary.scalar("MSE", mse) 
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) 

batch_size = 50
n_batchs = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batchs + batch_index) 
	indices = np.random.randint(m, size = batch_size) 
	X_batch = boston_plus_bias_normalized[indices]
	y_batch = boston.target.reshape(-1, 1)[indices]
	return X_batch, y_batch


init = tf.global_variables_initializer() 

with tf.Session() as sess:
	sess.run(init) 

	for epoch in range(n_epochs):
		for batch_index in range(n_batchs):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			if batch_index % 10 == 0:
				summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
				step = epoch + n_batchs + batch_index
				file_writer.add_summary(summary_str, step) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) 

	best_theta = theta.eval() 
print(best_theta)
file_writer.close() 