###Pro Deep learning chapter 2: Introduction to tensorflow:
##Tensorflow basics for development:
import tensorflow as tf 
import numpy as np 

#this activations an tensorflow interactive session like 
#in jypeter notebooks:
#tf.InteractiveSession() Note to self this command only works on juypter notebooks.

#define tensors: 
a = tf.zeros((2, 2))
b = tf.ones((2,2)) 

#sum the elements of the matrix across the horizontal axis 
init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run()#I believe there is no need to use this function since the two objects are not variables.
	c = tf.reduce_sum(b, reduction_indices = 1).eval() 
	print(c) 
#the following code snipet is not in the book but since I'm using sublime text for the 
#exercises I will have to use this method (note to self this method can be found in the second part of 
#hands on machine learning with scikit learn and tensorflow). 

#Checking the shape of the tensor:
print(a.get_shape()) 

#Reshape a tensor:
tf.reshape(a, (1,4))
with tf.Session() as sess:
	print(tf.reshape(a, (1,4)).eval())

tf.reset_default_graph() 

ta = tf.zeros((2,2)) 
print(ta) 
with tf.Session() as sess:
	print(ta.eval()) 
#You can only see the components within a tensor through the eval() tensorflow 
#function. Will need to look into how to create an interactive session without 
#juypter notebooks. 

#doing the same commands with the numpy library:
a = np.zeros((2,2))
print(a) 

#Define tensorflow constants:
tf.reset_default_graph() 

a = tf.constant(1) 
b = tf.constant(5) 
c = a * b

#Tensorflow Session for execution of the commands through run and eval.
with tf.Session() as sess:
	print(c.eval())
	print(sess.run(c))
#from what I can see the eval() and run() functions have the same output will need to look into 
#when it's apropriate to use eval() above run() and vice versa.

#Define tensorflow variables: 
w = tf.Variable(tf.ones(2,2), name = "weights") 

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) 
	print(sess.run(w)) 

#Just as I suspected tf.global_variables_initializer() is only needed if you 
#use a tf.Variable() object. 

#Define the Tensorflow variable with random initial values from standard normal distribution 
rw = tf.Variable(tf.random_normal((2,2)), name = "random_weights") 

#Invoke Session and display the initial state of the variable: 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(rw)) 

#Tensorflow Variable State Update 

tf.reset_default_graph() 

var_1 = tf.Variable(0, name = "var_1") 
add_op = tf.add(var_1, tf.constant(1))#This command adds a constant or variable to a 
#tensor.
upd_op = tf.assign(var_1, add_op) 

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run() 
	for i in range(5):
		print(sess.run(upd_op)) 


#display the tensorflow variable state:
x = tf.constant(1) 
y = tf.constant(5) 
z = tf.constant(7) 

mul_x_y = x * y 
final_op = mul_x_y + z 

with tf.Session() as sess:
	print(sess.run([mul_x_y, final_op]))

#Convert a numpy array to tensor:
a = np.ones((3,3)) 
b = tf.convert_to_tensor(a) 
with tf.Session() as sess:
	print(sess.run(b))

#Placeholders and feed dictionary:
inp1 = tf.placeholder(tf.float32, shape = (1,2))
inp2 = tf.placeholder(tf.float32, shape = (2,1))
output = tf.matmul(inp1, inp2)
with tf.Session() as sess:
	print(sess.run([output], feed_dict={inp1:[[1., 3.]], inp2:[[1], [3]]}))

##Optimizers in Tensorflow:
#GradientDescentOptimizer(), can be conceptualized as a optimized implemenation of the 
#basic full batch gradient descent method where the learning rate is not modified through the 
#training process. 
#look at page 130 for more information 
#Usage train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

#AdagradOptimizer: look at page 130 for the entire mathematical treatment of the method.
#With Adafrad, each parameter is updated with a different 
#learning rate. the sparser the feature is, the higher it 
#parameter update would be in an iteration. this is a good optimizer 
#to use in applications with natural language processing and image 
#processing where the data is sparse.
#Uses: train_op = tf.train.AdagradOptimizer(learning_rate=0.001, intial_accumulator_value = 0.1) 
#The initial_accumulator_value represents the initial non-zero normalizing factor for each weight. 

#RMSprop: RMSprop is the mini-batch version of the resilient backpropagation (Rprop) optimization technigue 
#that works best for full-batch learning. Rprop solves the issues of gradients' not pointing to the minmum 
#in cases where the cost function contours are elliptical. The special thing with 
#Rprop is that is doesn't use the magnitude of the gradients of the weight but only the signs in 
#determining how to update each weight. 
#Usage: train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10) 

#AdadeltaOptimizer: To see the full mathematical treatment look at page 134. 
#Usage: train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08) 
#Where rho represents gamma, epsilon represents e, and n represents the learning rate. 

#AdamOptimizer: page 135 
#Usage:training_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08) 

##XOR Implemenation Using Tensorflow: the author will only use a architecture with one hidden layer and a log-loss 
#output layer. 

#the input and output values:
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[4,2], name = "x") 
y = tf.placeholder(tf.float32, shape=[4,1], name = "y") 

#The weights for layers 1 and 2  
w1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name ="weights1") 
w2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "weights2") 

#The bias terms for layers 1 and 2 
b1 = tf.Variable(tf.zeros([2]), name = "Bias1") 
b2 = tf.Variable(tf.zeros([1]), name = "Bias2")

#define the final output through the forward pass:
z2 = tf.sigmoid(tf.matmul(X, w1) + b1) 
pred = tf.sigmoid(tf.matmul(z2, w2) + b2) 

#Define the cost function using the log-loss cost function formula.
cost = tf.reduce_mean(((y * tf.log(pred))+ ((1 - y) * tf.log(1.0 - pred))) * -1) 
learning_rate = 0.01 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

#the output and input values that will be used to train the model and placed in the placeholder function: 
XOR_X = [[0,0],[1,0],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer() 
sess = tf.Session() 
writer = tf.summary.FileWriter("./Downloads/XOR_logs", sess.graph_def) 

sess.run(init) 
for i in range(100000):
	sess.run(train_step, feed_dict={X: XOR_X, y: XOR_Y})

print("Final Prediction", sess.run(pred, feed_dict={X: XOR_X, y:XOR_Y})) 

##XOR logical implemenation using a linear activation function in place of the 
#sigmoid activation function from earlier:
X = tf.placeholder(tf.float32, shape = [4,2], name = "input") 
y = tf.placeholder(tf.float32, shape = [4,1], name = "output") 

w1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="weights1") 
w2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "weights2") 

b1 = tf.Variable(tf.zeros([2]), name = "bias1") 
b2 = tf.Variable(tf.zeros([1]), name = "bias2") 

z2 = tf.matmul(X, w1) + b1# As you can see the hidden layer does not have a sigmoid actionivation function 
#hence the output can be likened to that of a normal logistic regression.
pred = tf.sigmoid(tf.matmul(z2, w2) + b2) 

cost = tf.reduce_mean(((y * tf.log(pred)) + ((1 - y) * tf.log(1.0 - pred))) * -1) 
learning_rate = 0.01 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

XOR_X = [[0,0],[0,1],[1,0],[1,1]] 
XOR_Y = [[0],[1],[1],[0]] 

init = tf.global_variables_initializer() 
sess = tf.Session() 
sess.run(init) 
for i in range(10000):
	sess.run(train_step, feed_dict={X: XOR_X, y: XOR_Y}) 

print("Final Prediction", sess.run(pred, feed_dict={X: XOR_X, y: XOR_Y}))

##Linear regression using tensorflow:
#With this implementation the author talks about using linear regression with the 
#tensorflow package. The only difference between this implemenation and the one above is 
#that the cost function is conceptualized as the mean squared error in place of the cross entropy 
#cost function for the logistic regression. 

from sklearn.datasets import load_boston 

def read_infile():
	data = load_boston() 
	features = np.array(data.data) 
	target = np.array(data.target) 
	return features, target 

def feature_normalize(data):
	mu = np.mean(data, axis=0) 
	std = np.std(data, axis=0) 
	return (data - mu)/std
#This following function converts all the data points within each column into z-scores.

def append_bias(features, target):
	"""Append the feature for the bias term"""
	n_samples = features.shape[0]
	n_features = features.shape[1] 
	intercept_feature = np.ones((n_samples, 1))#Creates the bias column for the dataset.
	X = np.concatenate((features, intercept_feature), axis=1)#Interesting you can use concatenate to 
	#place a new column into the data matrix. Will need to test this out.
	X = np.reshape(X, [n_samples, n_features +1]) 
	y = np.reshape(target, [n_samples, 1]) 
	return X, y

#Execute the functions to read, normalize, and add append bias term to the data.
#these preprocessing steps can to streamlined with a pipeline for reproducibility.
tf.reset_default_graph() 

feature, target = read_infile() 
z_feature = feature_normalize(feature) 
X_input, y_input = append_bias(z_feature, target) 
num_features = X_input.shape[1] 

#Tensorflow components:
X = tf.placeholder(tf.float32, [None, num_features]) 
y = tf.placeholder(tf.float32, [None, 1]) 
w = tf.Variable(tf.random_normal((num_features, 1)), name = "Weights") 
init = tf.global_variables_initializer()

learning_rate = 0.01 
num_epoch = 1000 
cost_trace = [] 
pred = tf.matmul(X, w) 
error = pred - y 
cost = tf.reduce_mean(tf.square(error))#this is your mean squared error cost function,
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  

with tf.Session() as sess:
	sess.run(init) 
	for i in range(num_epoch):
		sess.run(train_op, feed_dict={X:X_input, y:y_input})
		cost_trace.append(sess.run(cost, feed_dict={X:X_input, y:y_input}))
	error = sess.run(error, {X:X_input , y:y_input}) 
	pred = sess.run(pred, {X:X_input}) 

print("MSE in training:", cost_trace[-1])#this gives back the final mean squared error value 
#for the model during the last iteration. 
#From what I can see (and in relation to his opening title) these lines of code are indeed 
#a perfect illustration of a vanilla linear regression model with no additional transformations.

#very nice the author used the cost_trace object to look at the learning curve of the model 
#without the use of the tensorboard server. 

#import matplotlib.pyplot as plt 
#plt.plot(range(num_epoch), cost_trace)
#It seems that I can't use matplotlib.pyplot at this moment. Will need to see what this is 
#all about at a later date.

#Plot the predicted house prices vs the actual house prices:
#fig, ax = plt.subplots() 
#plt.scatter(Y-input, pred) 
#ax.set_xlabel("actual house price") 
#ax.set_ylabel("predicted house price") 

##Multi-class classification with softmax function using full batch gradient descent:
from sklearn import datasets 
from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder

tf.reset_default_graph() 

def read_infile():
	mnist = fetch_mldata("MNIST original") 
	X_mnist, y_mnist = mnist["data"], mnist["target"]  
	X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)    
	return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = read_infile() 
print(X_train.shape) 
print(X_train) 
print(y_train.shape) 
print(y_train) 

#define the weights and biases for the neural network: 
def weights_biases_placeholder(n_dim):
	n_classes = 10 
	X = tf.placeholder(tf.float32, [None, n_dim]) 
	y = tf.placeholder(tf.int32, [None]) 
	w = tf.Variable(tf.random_normal([n_dim, n_classes], stddev=0.01), name="weights")
	b = tf.Variable(tf.random_normal([n_classes]), name="bias") 
	return X, y, w, b

def forward_pass(w, b, W):
	out = tf.matmul(X, w) + b
	return out 

#define the cost function for the softmax unit:
def multiclass_cost(out, y):
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))
	return cost 

#define the initialization op 
def init():
	return tf.global_variables_initializer() 

#define the training op 
def train_op(learning_rate, cost):
	op_train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
	return op_train

#The accuracy function used to see if the model is converging during training.
def evaluation(out, y):
	correct = tf.nn.in_top_k(out, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	return accuracy 

train_X, train_y, test_X, test_y = read_infile() 
X, y, w, b = weights_biases_placeholder(train_X.shape[1]) 
out = forward_pass(w, b, X) 
cost = multiclass_cost(out, y) 
accuracy = evaluation(out, y) 
learning_rate, epochs = 0.01, 200
op_train = train_op(learning_rate, cost) 
init = init() 
loss_trace = []
accuracy_trace = [] 

#Activate the tensorflow session and execute full batch gradient descent gradient descent:
with tf.Session() as sess: 
	sess.run(init) 

	for i in range(epochs):
		sess.run(op_train, feed_dict={X:train_X, y:train_y})
		loss = sess.run(cost, feed_dict={X:train_X, y:train_y})
		accuracy_train = sess.run(accuracy, feed_dict={X:train_X, y:train_y})  
		loss_trace.append(accuracy_train) 
		if (((i+1) >= 100) and ((i+1) % 100 == 0)):
			print("Epoch:", (i+1), "loss:", loss, "accuracy:", accuracy_train) 

	print("final training result:", loss, "accuracy:", accuracy_train) 
	loss_test = sess.run(cost, feed_dict={X:test_X, y:test_y}) 
	accuracy_test = sess.run(accuracy, feed_dict={X:test_X, y:test_y}) 
	print("test set results:","loss:",loss_test,"accuracy:",accuracy_test) 

import matplotlib.pyplot as plt
#finally cracked the problem. Needed to combine this author's coding style with that of 
#Hands on machine learning with scikit learn and tensorflow. At least now I know how to do 
#softmax logistic regression with the mnist dataset. Note to self, the onehot encoder 
#function doesn't work with this dataset's arrays. Will need to look into why this is.

##Multi-class classification with softmax function using stochastic gradient descent: 
tf.reset_default_graph() 

def weights_biases_placeholder(n_dim):
	n_classes = 10 
	X = tf.placeholder(tf.float32, [None, n_dim]) 
	y = tf.placeholder(tf.int32, [None]) 
	w = tf.Variable(tf.random_normal([n_dim, n_classes], stddev=0.01), name="weights")
	b = tf.Variable(tf.zeros([n_classes]), name="bias") 
	return X, y, w, b

def forward_pass(w, b, W):
	out = tf.matmul(X, w) + b
	return out 

def multiclass_cost(out, y):
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))
	return cost 
 
def init():
	return tf.global_variables_initializer() 

def train_op(learning_rate, cost):
	op_train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
	return op_train

def evaluation(out, y):
	correct = tf.nn.in_top_k(out, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	return accuracy 

train_X, train_y, test_X, test_y = read_infile() 
X, y, w, b = weights_biases_placeholder(train_X.shape[1]) 
out = forward_pass(w, b, X) 
cost = multiclass_cost(out, y) 
accuracy = evaluation(out, y) 
learning_rate, epochs = 0.01, 2#According to Andrew Ng stochastic gradient descent 
#training models only need two epoches to reach a respectible local minimum.
batch_size = 1000
num_batches = train_X.shape[0]/batch_size
op_train = train_op(learning_rate, cost) 
init = init() 
loss_trace = []
accuracy_trace = []

#Activate the tensorflow session and execute stochastic gradient descent:
with tf.Session() as sess: 
	sess.run(init) 

	for i in range(epochs):
		for j in range(train_X.shape[0]):
			X_batch, y_batch = train_X[j,], train_y[j] 
			sess.run(op_train, feed_dict={X:X_batch, y:y_batch})
			loss = sess.run(cost, feed_dict={X:X_batch, y:y_batch})
			accuracy_train = sess.run(accuracy, feed_dict={X:X_batch, y:y_batch})  
			loss_trace.append(accuracy_train) 
			if (((j+1) >= 100) and ((j+1) % 100 == 0)):
				print("index_position:", (j+1), "loss:", loss, "accuracy:", accuracy_train) 

	print("final training result:", loss, "accuracy:", accuracy_train) 
	loss_test = sess.run(cost, feed_dict={X:test_X, y:test_y}) 
	accuracy_test = sess.run(evaluation, feed_dict={X:test_X, y:test_y}) 
	print("test set results:","loss:",loss_test,"accuracy:",accuracy_test) 






