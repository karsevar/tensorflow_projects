###Hands on machine learning with scikit learn and tensorflow.
###chapter 9 Up and running with Tensorflow:
##creating your first graphic and running it in a session: 
import tensorflow as tf 

x = tf.Variable(3, name="x") 
y = tf.Variable(4, name = "y") 
f = x*x*y + y + 2
#These commands don't calculate anything. The practicioner will need to 
#initialize the code through the initializer method:

#illustration:
sess = tf.Session()
sess.run(x.initializer) 
sess.run(y.initializer) 
result = sess.run(f) 
print(result) 
sess.close()

#Easier implementation of this same initializer technique:
with tf.Session() as sess:
	x.initializer.run() 
	y.initializer.run() 
	result = f.eval() 
	print(result)

#Using an interactive session() alternative:
#sess = tf.InteractiveSession() 
#x.initializer.run() 
#result = f.eval() 
#print(result) 
#sess.close()
#Will need to look into how to run this code method correctly.

#(important) A tensorflow program is typically split into two parts: the first 
#part builds a computation graph (this is called the construction phase),
#and the second part runs it (this is called the execution phase). the construction 
#phase typically builds a computation graph representation the ML model 
#and the computations required to train it. The execution phase generally 
#runs a loop that evaluates a training step repeatedly

##Managing graphs:
x1 = tf.Variable(1) 
print(x1.graph is tf.get_default_graph()) 

#making multiple graphs:
graph = tf.Graph() 
with graph.as_default():
	x2 = tf.Variable(2) 

print(x2.graph is graph) 
print(x2.graph is tf.get_default_graph())#In this case the x2 graph is 
#not the default. Will need to see what the significance of this operation
#is.

##Lifecycle of a node value:
#When you evaluate a node, Tensorflow automatically determines the set of nodes 
#that it depends on and it evaluates these nodes first. For example, consider the following 
#code:
w = tf.constant(3) 
x = w+2 
y = x+5
z = x*3 

with tf.Session() as sess:
	print(y.eval())
	print(z.eval())

#All node values are dropped between graph runs, except variable values, which
#are maintained by the session across graph runs (queues and readers also maintain
#some state). a variable starts its life when its initializer is run, and 
#it ends when the session is closed. 

#If you want to evaluate y and z efficiently, without evaluating w and x twice 
#as in the previous code, you must ask tensorflow to evaluate y and z in just 
#one graph run. 
with tf.Session() as sess:
	y_val, z_val = sess.run([y, z]) 
	print(y_val)
	print(z_val) #this alternative is more computationally efficient.

##Linear Regression with Tensorflow:
#tensorflow operations are called ops and multiplication and addition ops 
#takes two inputs to produce one output while constant and variable ops takes 
#no inputs. 

#With this example the author uses the housing dataset from chapter 2:
import numpy as np 
from sklearn.datasets import fetch_california_housing
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.datasets import load_boston
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler  

housing = pd.read_csv("/Users/masonkarsevar/housing.csv")
housing = housing.drop("ocean_proximity", axis = 1)
housing_data = housing.drop("median_house_value", axis = 1)
housing_data = list(housing_data)
housing_target = housing["median_house_value"]

class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names 
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values 

pipeline = Pipeline([
	("selector", DataFrameSelector(housing_data)),
	("imputer", Imputer(strategy = "mean")),
]) 

X = pipeline.fit_transform(housing)
print(X)
y = housing_target 

m, n = X.shape 
housing_plus_bias = np.c_[np.ones((m, 1)), X]  
X = tf.constant(housing_plus_bias, dtype = tf.float32, name = "X") 
y = tf.constant(y.values.reshape(-1, 1), dtype = tf.float32, name = "y") 
XT = tf.transpose(X) 
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
	theta_value = theta.eval()

print(theta_value)

boston = load_boston()
m, n = boston.data.shape 
boston_data_plus_bias = np.c_[np.ones((m, 1)), boston.data]
X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y") 
XT = tf.transpose(X) 
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
	theta_value = theta.eval()
print(theta_value)#since the california dataset was not working. I was 
#forced to use the boston dataset instead. the problem with this is that 
#I imported the california dataset as a panda and if pandas and tensorflow 
#data structures are not compatible I will need to find a different read_csv()
#function library. 

#Found out what the problem was. It seems that the na values were getting 
#in the way to the normal equation computation. For the california dataset 
#I was forced to write a pipeline that imputes all the na values within the 
#dataset's variable portion. The values are a bit different from what the 
#author obtained. I will need to see if this is due to my interpretation 
#of the normal equation or the imputation method I used. 

lin_reg = LinearRegression() 
lin_reg.fit(boston.data, boston.target.reshape(-1, 1))
print(np.r_[lin_reg.intercept_.reshape(-1,1), lin_reg.coef_.T]) 
#Sweet the normal equation calculated the same theta values as the 
#sklearn LinearRegression() function.

##Implementing Gradient Descent:
pipeline = Pipeline([
	("selector", DataFrameSelector(housing_data)),
	("imputer", Imputer(strategy = "mean")),
	("std_scaler", StandardScaler())
])
X = pipeline.fit_transform(housing)
y = housing["median_house_value"]
m, n = X.shape 

##Manually Computing the Gradients:
#California housing dataset:
n_epoches = 1000
learning_rate = 0.01 
housing_plus_bias_scaled = np.c_[np.ones((m, 1)), X] 

X = tf.constant(housing_plus_bias_scaled, dtype = tf.float32, name = "X")
y = tf.constant(y.values.reshape(-1, 1), dtype = tf.float32, name = "y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions") 
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
gradients = 2/m * tf.matmul(tf.transpose(X), error) 
training_op = tf.assign(theta, theta - learning_rate * gradients) 

init = tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epoches):
 		if epoch % 100 == 0:
 			print("Epoch", epoch, "MSE =", mse.eval())
 		sess.run(training_op)
	best_theta = theta.eval()

print(best_theta) 

#Boston housing dataset:
m, n = boston.data.shape 
boston_data_plus_bias = np.c_[np.ones((m, 1)), boston.data]
scaler = StandardScaler() 
scaler.fit(boston.data) 
boston_data = scaler.transform(boston.data) 
boston_data_plus_bias = np.c_[np.ones((m, 1)), boston_data]

n_epoches = 10000
learning_rate = 0.02 

X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = "X") 
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
	for epoch in range(n_epoches):
 		if epoch % 100 == 0:
 			print("Epoch", epoch, "MSE =", mse.eval())
 		sess.run(training_op)
	best_theta = theta.eval()

print(best_theta)#Again this dataset is less convoluted than the california 
#housing dataset. Still though the mse values are oddly large will need to look into
#this problem. 

##Using autodiff:
def my_func(a, b):
	z = 0 
	for i in range(100):
		z = a * np.cos(z + i) + np.sin(b - i) 
	return z

#to create the autodiff function within the gradient descent algorithm 
#above all you need to do is replace the gradients = 2/m * tf.matmul(tf.transpose(X), error)
#call with gradients = tf.gradients(mse, [theta])[0] 

#Let's try this out with the Boston dataset:
m, n = boston.data.shape 
boston_data_plus_bias = np.c_[np.ones((m, 1)), boston.data]
scaler = StandardScaler() 
scaler.fit(boston.data) 
boston_data = scaler.transform(boston.data) 
boston_data_plus_bias = np.c_[np.ones((m, 1)), boston_data]

n_epoches = 10000
learning_rate = 0.02 

X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = "X") 
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
	for epoch in range(n_epoches):
 		if epoch % 100 == 0:
 			print("Epoch", epoch, "MSE =", mse.eval())
 		sess.run(training_op)
	best_theta = theta.eval()

print(best_theta)
#I saddly don't see the difference between this and the other method.
#will need to experiment with this futher. 

#to look at the different autodiff methods look at page 239. 

##Using an optimizer:
#Gradient descent optimizer:
n_epoches = 10000
learning_rate = 0.02 

X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions") 
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(mse)  

init = tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epoches):
 		if epoch % 100 == 0:
 			print("Epoch", epoch, "MSE =", mse.eval())
 		sess.run(training_op)
	best_theta = theta.eval()

print(best_theta)

#momentum optimizer:
n_epoches = 10000
learning_rate = 0.02 

X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions") 
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,
	momentum = 0.9) 
training_op = optimizer.minimize(mse)  

init = tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epoches):
 		if epoch % 100 == 0:
 			print("Epoch", epoch, "MSE =", mse.eval())
 		sess.run(training_op)
	best_theta = theta.eval()

print(best_theta)#the author is right the momentumoptimizer() 
#method was by far the fastest.

##Feeding Data to the training algorithm:
# To feed data into the algorithm you need to use the placeholder() function
#node and specify the output tensor's data type. 
A = tf.placeholder(tf.float32, shape=(None, 3))# this creates an output 
#of type float32 and shape two dimensions with 3 columns.
B = A + 5 
with tf.Session() as sess:
	B_val_1 = B.eval(feed_dict={A: [[1,2,3]]}) 
	B_val_2 = B.eval(feed_dict = {A: [[4,5,6], [7,8,9]]})

print(B_val_1) 
print(B_val_2) 

##Creating a mini_batch gradient descent method:
n_epoches = 10000
learning_rate = 0.02 

X = tf.placeholder(tf.float32, shape = (None, n + 1), name = "X") 
y = tf.placeholder(tf.float32, shape = (None, 1), name = "y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions") 
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(mse)

n_epoches = 10 
batch_size = 100 
n_batches = int(np.ceil(m / batch_size)) 

def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batches + batch_index) 
	indices = np.random.randint(m, size = batch_size) 
	X_batch = boston_data_plus_bias [indices] 
	y_batch = boston.target.reshape(-1, 1)[indices] 
	return X_batch, y_batch   

init = tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epoches):
 		for batch_index in range(n_batches):
 			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
 			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

	best_theta = theta.eval()

print(best_theta)

##Saving and Restoring Models:
n_epoches = 10000
learning_rate = 0.02 

X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = "X") 
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions") 
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error), name = "mse") 
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,
	momentum = 0.9) 
training_op = optimizer.minimize(mse)  

init = tf.global_variables_initializer() 
#saver = tf.train.Saver() 

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epoches):
 		if epoch % 100 == 0:
 			#save_path = saver.save(sess, "/tmp/my_model_final.ckpt")#Sweet 
 			#this actually works. 
 			print("Epoch", epoch, "MSE =", mse.eval())
 		sess.run(training_op)
	best_theta = theta.eval()

print(best_theta)

#restoring the values from the last session:
#init = tf.global_variables_initializer() 
#saver = tf.train.Saver() 

#with tf.Session() as sess:
	#saver.restore(sess, "/tmp/my_model_final.ckpt") 
	#best_theta_restored = theta.eval()
	#for epoch in range(n_epoches):
 		#if epoch % 100 == 0:
 			#save_path = saver.save(sess, "/tmp/my_model_final.ckpt")#Sweet 
 			#this actually works. 
 			#print("Epoch", epoch, "MSE =", mse.eval())
 		#sess.run(training_op)
	#best_theta = theta.eval()

#print(best_theta)

##Visualizing the graph and training curves using tensorboard:
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")#cool a regular expression!!!
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now) #this time stamps the mse data 
#the gradient descent algorithm spits out.

n_epochs = 1000
learning_rate = 0.02 
m, n = boston.data.shape 

X = tf.placeholder(tf.float32, shape = (None, n + 1), name = "X") 
y = tf.placeholder(tf.float32, shape = (None, 1), name = "y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed = 42), name = "theta") 
y_pred = tf.matmul(X, theta, name = "predictions")

with tf.name_scope("loss") as scope:
	error = y_pred - y 
	mse = tf.reduce_mean(tf.square(error), name = "mse") 

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar("MSE", mse) 
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10 
batch_size = 100
n_batches = int(np.ceil(m / batch_size)) 

def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batches + batch_index) 
	indices = np.random.randint(m, size = batch_size) 
	X_batch = boston_data_plus_bias[indices] 
	y_batch = boston.target.reshape(-1, 1)[indices] 
	return X_batch, y_batch  

with tf.Session() as sess: 
	sess.run(init) 

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			if batch_index % 10 == 0:
				summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
				step = epoch * n_batches + batch_index 
				file_writer.add_summary(summary_str, step) 
			sess.run(training_op, feed_dict={X: X_batch, y:y_batch})

	best_theta = theta.eval()

file_writer.flush()
file_writer.close()
print("Best theta:")
print(best_theta) 
print(error.op.name)
print(mse.op.name)
#I finally got the dashboard to work!!! 
#Will need to experiment a little more with the namespace command name_scope().

##Modularity :
#Creating a ReLU linear function model with tensorflow:
n_features = 3
X = tf.placeholder(tf.float32, shape = (None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name = "weight1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name = "weight2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name = "bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name = "z1")
z2 = tf.add(tf.matmul(X, w2), b2, name = "z2")

relu1 = tf.maximum(z1, 0., name = "relu1")
relu2 = tf.maximum(z2, 0., name = "relu2")

output = tf.add(relu1, relu2, name = "output")

#Stream lined tensorflow function of this same computation using add_n()
def relu(X):
	w_shape = (int(X.get_shape()[1]), 1)
	w = tf.Variable(tf.random_normal(w_shape), name = "weights")
	b = tf.Variable(0.0, name = "bias")
	z = tf.add(tf.matmul(X, w), b, name = "z")
	return tf.maximum(z, 0., name = "relu")

n_features = 3 
X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name = "output") 

##Shared Variables:
#If you want to share a variable between components of your graph, one simple 
#option is to create it first, then pass it as a parameter to the functions that need it.
#illustration:
def relu(X, threshold):
	with tf.name_scope("relu"):
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal(w_shape), name = "weights")
		b = tf.Variable(0.0, name = "bias")
		z = tf.add(tf.matmul(X, w), b, name = "z")
		return tf.maximum(z, 0., name = "relu")


threshold = tf.Variable(0.0, name = "threshold")
X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name = "output")

#The problem with this method is that programmers will have to input 
#variables into the relu tensorflow call all the time. A way to 
#get around this is to pass the threshold argument a dictionary of values 
#or create a relu function:

def relu(X):
	with tf.name_scope("relu"):
		if not hasattr(relu, "threshold"):
			relu.threshold = tf.Variable(0.0, name = "threshold")
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal(w_shape), name = "weights")
		b = tf.Variable(0.0, name = "bias")
		z = tf.add(tf.matmul(X, w), b, name = "z")
		return tf.maximum(z, relu.threshold, name = "max")

X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X") 
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name = "output") 

#Using the get_variable() alternative:
#with tf.variable_scope("relu"):
	#threshold = tf.get_variable("threshold", shape=(),
		#initializer=tf.constant_initializer(0.0))
#If you want to reuse this specific variable all you need to do is set the 
#reuse argument to True (you don't need to define the shape() or the 
#initializer arguments). 

#Once reuse is set to True, it cannot be set back to False within the block.
#moreover, if you define other variable scopes inside this one, they will 
#inherit reuse = True.Only variables created by get_variable() can be reused 
#this way. 

#now the entire code:
def relu(X):
	with tf.variable_scope("relu", reuse=True): 
		threshold = tf.get_variable("threshold")
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal(w_shape), name = "weights")
		b = tf.Variable(0.0, name = "bias")
		z = tf.add(tf.matmul(X, w), b, name = "z")
		return tf.maximum(z, threshold, name = "max")

X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X") 
with tf.variable_scope("relu"):
	threshold = tf.get_variable("threshold", shape=(), 
		initializer = tf.constant_initializer(0.0))

relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name = "output")
file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close() 

#The same code as the one above except the threshold variable is 
#partitioned within the relu function call assembly.
def relu(X):
	threshold = tf.get_variable("threshold", shape = (), initializer=tf.constant_initializer(0.0))
	w_shape = (int(X.get_shape()[1]), 1)
	w = tf.Variable(tf.random_normal(w_shape), name = "weights")
	b = tf.Variable(0.0, name = "bias")
	z = tf.add(tf.matmul(X, w), b, name = "z")
	return tf.maximum(z, threshold, name = "max")

X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X") 
relus = []
for relu_index in range(5):
	with tf.variable_scope("relu", reuse = (relu_index >=1)) as scope:
		relus.append(relu(X))
output = tf.add_n(relus, name = "output")
file_writer = tf.summary.FileWriter("logs/relu7", tf.get_default_graph())
file_writer.close() 













