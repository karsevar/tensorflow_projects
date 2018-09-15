###Hands on machine learning with Scikit learning and tensorflow
###chapter 10 exercises:

##1.) For this problem look that the tensorflow notes notebook 
#entry June 19, 2018.

##2.) The initial problem with the classic implementation of Perceptron
#neural networks (a single layer of linear threshold units trained using 
#the Perceptron training method) is that they are incapable of solving 
#some trivial problems (namely the exclusive or (XOR) classification problem.
#Today we found out that is could be fix through simply stacking multiple 
#perceptrons (thus creating multi-layer perceptrons). In other words, 
#this problem was fixed through adding additional hidden layers into the 
#perceptron architecture. As such this showed researchers that neural 
#networks can predict more complex outputs through simply increasing the 
#number of hidden layers within the network.

##3.) The logistic activation function was a huge leap forward in training 
#MLPs because before researchers were forced to use the standard step wise 
#activation function to initialize classic perceptron neural networks. the problem 
#with using the step activation function (in place of the logistic function) is 
#that training was more or less impossible to carry out with gradient descent 
#(since within the step function, there is no gradient for the algorithm to work 
#with a minimize). And so since the logistic function is convex in nature,
#the gradient descent algorithm was able to train neural networks through 
#minimizing a defined cost function. This discovery is still being used 
#modern neural network applications (regardless of current scholarship in this area 
#stating that the ReLU activation function is actually the most efficient).

##4.) The three most popular activation functions are:

#The hyperbolic tangent function: Just like the logistic function 
#it is S-shaped, continuous, and differentiable, but its output value ranges from -1 
#to 1 (instead of 0 to 1 in the case of logistic function), which tends to make 
#each layer's output more or less normalized at the beginning of training. 

#The ReLU function: ReLU(z) = max(0, z). It is continuous but unfortunately not differentiable
#at z = 0 (the slope changes abruptly, which can make gradient descent bounce around). However, 
#in practice it works very well and has the advantage of being fast to compute. 
#Most importantly, the fact that it does not have a maximum output value also 
#helps reduce some issues during gradient descent.

#The logistic function sigmoid(z) = 1 / 1 + exp(z). This function is used 
#as the default shape for the logistic regression algorithm as well as other 
#implementations of linear decision boundary machine learning methods. The upper 
#limit is 1, the lower limit is 0, and the middle of the function is 0.5.
#The shape is very similar to the hyperbolic tangent function. And lastly of the 
#three activation functions, the logistic function is by far the most popular in that 
#it perfectly mirrors the neural activation curve within biological neural networks. 

##5.) Hypothetic situation: 10 passthrough neurons, 50 neurons in the hidden layer, 
#and 3 total neurons within the output layer. 

#With that said, I can assume that the dataset (being inputted in the neural network 
#within the first layer) has a total of 10 variables and an unknown amount of observations.
#the hidden layer (with a total of 50 neurons) will create a matrix with 50 columns and an 
#unknown amount of rows (since we don't know how any observations are in the dataset). And the 
#output layer will have a 3 by 1 vector for each output (since the neural network) is 
#solving for a three class classification problem. 

#Author's answer: 
#the shape of the input matrix X is m by 10, where m represents the training batch size.

#The shape of the hidden layer's weight vector W_h is 10 by 50 and the length of 
#its bias vector b_h is 50.

#The shape of the hidden layer's weight vector W_0 is 50 by 3, and the length of its bias 
#becotr b_0 is 3.

#The shape of the network's output matrix Y is m by 3.

#Y = ReLU(ReLU(X * W_h + b_h) * W_0 + b_0). Recall that the ReLU function just sets
#every negative number in the matrix to zero. also note that when you are adding a bias 
#vector to a matrix, it is added to every single row in the matrix which is 
#called broadcasting. 

##6.) 
#For a email classification model that only has two respective classes
#(spam or ham) I can say that the output layer will have a total of two 
#neurons.
#As for the activation function, I believe that softmax logistic regression 
#might be a little excessive (since softmax logistic regression is only used for 
#multiclass classification problems). Most likely all you need to do is place the 
#output logit into a sigmoid function (1 / 1 + exp(-z)) and use the argmax function.

#For the mnist classification dataset, the number of output layers should be 10 
#since there are a total of 10 different classes within the target column. 
#As for the activation function, softmax logistic activation function (since 
#there are a total of 10 output classes and the logistic activation function 
#can only handle two output classes).

#for the California housing price dataset, the best number of output neurons is 
#one (since I believe that continuous output variables only require one output neuron) 
#According to the author, continuous output variables don't need an activation 
#function since the logit itself is the output value.  

##7.) 
#Author's answer:
#Backpropagation is a technique used to train artificial neural networks. I first computes the 
#gradients of the cost function with regards to every model parameter (all the weights 
#and biases), and then it performs a gradient descent step using these gradients 
#This backpropagation step is typically performed thousands of millions of times 
#using 

##9.) 
import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets("MNIST_data/", one_hot = False)  
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

def neuron_layer(X, n_neurons, name, activation = None):
	with tf.name_scope("name"):
		n_inputs = int(X.get_shape()[1]) 
		stddev = 2 / np.sqrt(n_inputs) 
		init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev) 
		W = tf.Variable(init, name = "kernel") 
		b = tf.Variable(tf.zeros([n_neurons]), name = "bias")#Here is the 
		#bias term. For neural networks the bias term is a neuron.
		Z = tf.matmul(X, W) + b 
		if activation is not None:
			return activation(Z) 
		else:
			return Z 

with tf.name_scope("dnn"):
	hidden1 = neuron_layer(X, n_hidden1, name = "hidden1",
		activation=tf.nn.relu) 
	hidden2 = neuron_layer(hidden1, n_hidden2, name = "hidden2",
		activation=tf.nn.relu) 
	logits = neuron_layer(hidden2, n_outputs, name = "outputs")

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

n_epochs = 60 
batch_size = 50
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid) 







