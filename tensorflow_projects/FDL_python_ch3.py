###Fundamentals of deep learning chapter 3:
import tensorflow as tf 
import numpy as np 
#from read_data import get_minibatch

weights = tf.Variable(tf.random_normal([300, 200], stddev = 0.5), name = "weights")
print(weights)#interesting, I will need to read more about tensors as I 
#don't understand the following output. 

#This is a variable that was created in the tensflow framework that 
#describes the weights connecting neurons between two layers of a feed 
#forward neural network. 

#The random_normal() function is used to create a normal distribution 
#that connects a layer containing 300 neurons to 200 neurons with 
#a standard deviation of 0.5. 

#common tensors from the TensorFlow API documentation:
#tf.zeros(shape, dtype = tf.float32, name = None) 
#tf.ones(shape, dtype = tf.float32, name = None) 
#tf.random_normal(shape, mean = 0.0, stddev = 1.0, dtype = tf.float32, 
	#seed = None, name = None) 
#tf.truncated_normal(shape, mean = 0.0, stddev = 1.0, dtype = tf.float32,
	#seed = None, name = None)#I wonder if this is a skewed distribution 
#generator. Will need to look into this. 
#tf.random_uniform(shape, minval = 0, maxval = None, dtype = tf.float32, 
	#seed = None, name = None)#Most likely this function creates uniform 
#distributions.  

##Placeholder:
#populating the model with X variables every time the graph is ran.
x = tf.placeholder(tf.float32, name = "x", shape = [None, 784])#Used as the 
#X variable populator.
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name = "W")#can be 
#conceptualized as the theta values within a neural network. 
multiply = tf.matmul(x, W)# matrix multiplication command in tensorflow.

##Sessions in Tensorflow:
x = tf.placeholder(tf.float32, name = "x", shape = [None, 784]) 
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name = "W")
b = tf.Variable(tf.zeros([10]), name = "biases")
output = tf.matmul(x, W) + b 
init_op = tf.initialize_all_variables()

sess = tf.Session() 
sess.run(init_op) 
#feed_dict = {"x": get_minibatch()}
#sess.run(output, feed_dict=feed_dict)  
#there is a bug within the read_data module will look into how to fix 
#this problem later.

##Navigating Variable Scopes and sharing variables:
def my_network(input):
	W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1), name = "W_1")
	b_1 = tf.Variable(tf.zeros([100]), name = "biases_1")
	output_1 = tf.matmul(input, W_1) + b_1

	W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1), name = "W_2")

	b_2 = tf.Variable(tf.zeros([50]), name = "biases_2") 
	output_2 = tf.matmul(output_1, W_2) + b_2 

	W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1), name = "W_3") 
	b_3 = tf.Variable(tf.zeros([10]), name = "biases_3") 
	output_3 = tf.matmul(output_2, W_3) + b_3 

	print("Printing names of weight parameters\n")
	print(W_1.name, W_2.name, W_3.name) 
	print("Printing names of bias parameters")
	print(b_1.name, b_2.name, b_3.name) 

	return output_3 

i_1 = tf.placeholder(tf.float32, [1000, 784], name = "i_1") 
my_network(i_1)# Sweet I have the same outputs as the author. this is 
#really promising.

i_2 = tf.placeholder(tf.float32, [1000, 784], name = "i_2")
my_network(i_2)#Interesting the author was right the names are indeed 
#different. 

#We want the model to reuse the training examples not create a new model 
#during each training session. Hence variable_scope and get_variable should 
#be used. 

def layer(input, weight_shape, bias_shape):
	weight_init = tf.random_uniform_initializer(minval=-1, maxval = 1) 
	bias_init = tf.constant_initializer(value = 0) 
	W = tf.get_variable("W", weight_shape, initializer = weight_init) 
	b = tf.get_variable("b", bias_shape, initializer = bias_init) 
	return tf.matmul(input, W) + b

def my_network(input):
	with tf.variable_scope("layer_1"):
		output_1 = layer(input, [784, 100], [100]) 

	with tf.variable_scope("layer_2"):
		output_2 = layer(output_1, [100, 50], [50]) 

	with tf.variable_scope("layer_3"):
		output_3 = layer(output_2, [50, 10], [10]) 

	return output_3

i_1 = tf.placeholder(tf.float32, [1000, 784], name = "i_1") 
my_network(i_1)
i_2 = tf.placeholder(tf.float32, [1000, 784], name = "i_2")
#my_network(i_2) 
#This returned an error since according to the author the get_variable 
#tensorflow command checks that a variable of the given name hasn't been instantiated
#By default, sharing is not allowed (just to be safe). You can program 
#the variable scope to share through an explicit command. 

with tf.variable_scope("shared_variables") as scope: 
	i_1 = tf.placeholder(tf.float32, [1000, 784], name = "i_1")
	my_network(i_1) 
	scope.reuse_variables() 
	i_2 = tf.placeholder(tf.float32, [1000, 784], name = "i_2") 
	my_network(i_2) 

tf.reset_default_graph() 

##Managing Models over the CPU and GPU:
#Will need to come back to this section later since pages 51 to 52 tells the 
#reader how to implement a multinode machine learning system without the 
#need to azure and other cloud computing platforms. 

##Specifying the logistic regression model in tensorflow:
#In this example the author will use the MNIST dataset to train a neural 
#network model that is trained with logistic regression. Interestingly 
#since the author is using a neural network with the logistic regression
#model he doesn't have to use the allvsone or onevsone methodology. 

import time, shutil, os 
from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 
import numpy as np 

mnist = fetch_mldata("MNIST original")
X_mnist, y_mnist = mnist["data"], mnist["target"] 
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2,
	random_state = 42, shuffle = True)
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train = y_train.astype(np.float32) 
y_test = y_test.astype(np.float32) 
X_train, X_valid = X_train[5000:], X_train[:5000] 
y_train, y_valid = y_train[5000:], y_train[:5000] 

n_inputs = 28 * 28 
n_outputs = 10 
learning_rate = 0.01 
training_epoches = 1000
batch_size = 100 
display_step = 1 
n_batches = int(np.ceil(len(X_train)/batch_size)) 

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch


def inference(X):
	init = tf.constant_initializer(value = 0) 
	W = tf.get_variable("W", [784, 10], initializer=init) 
	b = tf.get_variable("b", [10], initializer =init) 
	output = tf.nn.softmax(tf.matmul(X, W) + b) 
	return output 

def loss(output, y):
	dot_product = y * tf.log(output) 
	xentropy = -tf.reduce_sum(dot_product, reduction_indices = 1)
	loss = tf.reduce_mean(xentropy) 
	return loss 

def training(cost, global_step):
	optimizer = tf.train.GradientDescentOptimizer(
		learning_rate)
	train_op = optimizer.minimize(cost, global_step=global_step) 
	return train_op 

def evaluate(output, y):
	correct_prediction = tf.nn.in_top_k(output, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.int32))
	return accuracy 

#the following is a logistic regression tensorflow setup. Will need to 
#learn more about the cross entropy part of the module. 

##Logging and training the logistic regression model: 
def training(cost, global_step):
	tf.summary.scalar("cost", cost) 
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	train_op = optimizer.minimize(cost, global_step=global_step) 
	return train_op 

X = tf.placeholder(tf.float32, shape = (None, 784), name = "X") 
y = tf.placeholder(tf.float32, shape = (None), name = "y")

output = inference(X) 
cost = loss(output, y) 
global_step = tf.Variable(0, name = "global_step", trainable = False) 
training_op = training(cost, global_step) 
eval_op = evaluate(output, y)  
saver = tf.train.Saver() 
summary_writer = tf.summary.FileWriter("logistic_logs/", graph_def = sess.graph_def) 

init = tf.global_variables_initializer()

with tf.Session() as sess: 
	sess.run(init)

	for epoch in range(training_epoches):
		avg_cost = 0. 
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
			#compute average loss
			minibatch_cost = sess.run(cost, feed_dict = {X:X_batch, y:y_batch}) 
			avg_cost += minibatch_cost/n_batches

		#display logs per epoch step.
		if epoch % display_step == 0:
			accuracy = sess.run(eval_op, feed_dict = {X:X_valid, y:y_valid}) 

			print("Validation Error:", (1 - accuracy))  
			saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step) 

	print("optimization finished") 

	accuracy = sess.run(eval_op, feed_dict = {X:X_test, y:y_test}) 

	print("Test Accuracy:", accuracy) 

#really there's just too many errors to deal with when attempting to apply this 
#implementation. I guess I'm forced to use the hands on machine learning with sklearn 
#and tensorflow logistic regression model. this is rather sad.

tf.reset_default_graph()
#I really don't understand what the author is getting at with this computation 
#map it's a little convoluted for my taste. 
#computation map from page 56 without the with tf.Graph().as_default(): command.




