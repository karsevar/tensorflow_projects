###Hands on machine learning with scikit learn and tensorflow.
###chapter 9 Up and running with Tensorflow:
#experiments: 
import tensorflow as tf

x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = "y") 
f = x * x * y + y + 2 
with tf.Session() as sess:
	x.initializer.run() 
	y.initializer.run() 
	result = f.eval() 
print(f)# Now I understant the f, x, and y objects are only tensors while 
#the result object is actually the values after running the computation.
print(result)

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run() 
	result = f.eval()
#Does the same thing as the initialization with block above except with less 
#code.

graph = tf.Graph() 
with graph.as_default():
	x2 = tf.Variable(2) 

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())#the response is false. Now I 
#Understand why all of my other exercises were placed into one large compuational 
#graph. I hope the computations weren't all combined together as a result.
print(tf.get_default_graph())
#Important command to reset the graph session after every computation:
tf.reset_default_graph()

##Lifecycle of a node value:
w = tf.constant(3)
x = w + 2 
y = x +5 
z = x * 3 
with tf.Session() as sess:
	y_val, z_val = sess.run([y, z]) 
	print(y_val)
	print(z_val) 

tf.reset_default_graph() 

##Linear Regression with Tensorflow: 
from sklearn.datasets import load_boston
import numpy as np  

boston = load_boston() 
m, n = boston.data.shape
print(boston.target.reshape(-1, 1)) 

boston_plus_bias = np.c_[np.ones((m, 1)), boston.data]

X = tf.constant(boston_plus_bias, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y") 
XT = tf.transpose(X) 
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y) 

with tf.Session() as sess:
	theta_value = theta.eval() 

print(theta_value) 
tf.reset_default_graph()

##Implementing Gradient Descent:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
scaler.fit(boston_plus_bias) 
boston_plus_bias_normalized = scaler.transform(boston_plus_bias) 

n_epochs = 10000 
learning_rate = 0.03

X = tf.constant(boston_plus_bias_normalized, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
gradients = 2/m * tf.matmul(tf.transpose(X), error) 
training_op = tf.assign(theta, theta - learning_rate * gradients) 

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	sess.run(init) 

	for epoch in range(n_epochs):
		sess.run(training_op) 

	best_theta = theta.eval() 
print(best_theta) 

tf.reset_default_graph() 

##Gradient descent using autodiff:
X = tf.constant(boston_plus_bias_normalized, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
gradients = tf.gradients(mse, [theta])[0] 
training_op = tf.assign(theta, theta - learning_rate * gradients) 

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	sess.run(init) 

	for epoch in range(n_epochs):
		sess.run(training_op) 

	best_theta = theta.eval() 
print(best_theta)

tf.reset_default_graph() 

##Gradient Descent optimizer:
X = tf.constant(boston_plus_bias_normalized, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	sess.run(init)   

	best_theta = theta.eval() 
print(best_theta)

tf.reset_default_graph() 

A = tf.placeholder(tf.float32, shape = (None, 3))
B = A + 5 
with tf.Session() as sess:
	B_val_1 = B.eval(feed_dict={A: [[1,2,3]]})
	B_val_2 = B.eval(feed_dict={A: [[4,5,6], [7,8,9]]})

print(B_val_1) 
print(B_val_2) 

tf.reset_default_graph() 

##mini_batch gradient descent with summary tensorboard extensions:
from datetime import datetime 

now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

X = tf.placeholder(tf.float32, shape = (None, n + 1), name = "X")# the shape argument 
#has a plus one value because it is counting the bias vector which wasn't defined before 
#calling the m and n values.  
y = tf.placeholder(tf.float32, shape = (None, 1), name = "y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions")# This part can be explained by 
#the mathematical expression H(theta_j) or rather theta^transpose*X.

with tf.name_scope("loss") as scope:
	error = y_pred - y# The mathematical expression for this is sum(H(theta) - y)
	mse = tf.reduce_mean(tf.square(error), name = "mse")#the mathematical expression for this is 
	#sum(H(theta) - y)^2. I find it interesting that the author didn't calculate this 
	#value through the Andrew Ng interpretation sum(H(theta) - y)X^i_j.  

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(mse)
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

tf.reset_default_graph() 

##relu exercise in modularity: 
def relu(X):
	with tf.variable_scope("relu", reuse = True):
		threshold = tf.get_variable("threshold") 
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal(w_shape), name = "weights") 
		b = tf.Variable(0.0, name = "bias") 
		z = tf.add(tf.matmul(X, w), b, name = "z") 
		return tf.maximum(z, threshold, name = "max") 
n_features = 3
X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X")
with tf.variable_scope("relu"):
	threshold = tf.get_variable("threshold", shape = (),
		initializer = tf.constant_initializer(0.0))
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name = "output")
file_writer = tf.summary.FileWriter("logs/relu4", tf.get_default_graph()) 
file_writer.close()

tf.reset_default_graph()

###Fundamentals of Deep learning chapter 3: 












