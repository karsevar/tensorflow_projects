###Hands on machine learning with tensorflow and scikit learn:
###Chapter 10 Introduction to Artificial Neural Networks
##The Perceptron:
#The Perceptron is one of the simplest ANN architectures. It is based on a slightly
#different artificial neuron called a linear threshold unit: The inputs 
#and output are now numbers and each input connection is associated with a weight 
#The LTU computes a weighted sum of its inputs, then applies a step function 
#to the sum and outputs the results

#A single LTU can be used for simple linear binary classification. It 
#computes a linear combination of the inputs and if the result exceeds 
#a threshold, it outputs the positive class or else outputs the negative class.
#Much like gradient descent, training a LTU neural network revolves around 
#finding the best W_0, w_1, w_2 values for the dataset. 

#Training algorithm (Hebbian learning): The perceptron is fed one training 
#instance at a time, and for each instance it makes it predictions. For 
#every output neuron that produced a wrong prediction, it reinforces the connection 
#weights from the inputs that would have contributed to the correct 
#prediction.
#Look at page 259 for the equation.

#The decision boundary of each output neuron is linear, so Perceptrons are incapable 
#of learning complex patterns. 

##Creating a single LTU network using the Perceptron class:
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron 

iris = load_iris() 
X = iris.data[:, (2,3)]#petal length, petal width  
y = (iris.target == 0).astype(np.int)
per_clf = Perceptron(random_state = 42) 
per_clf.fit(X, y) 
y_pred = per_clf.predict([[2, 0.5]])# Predicts the class of an iris with a 
#petal length of 2 and petal width of 0.5. 
print(y_pred)

#the SGDClassifier can perform the same computation through setting 
#the hyperparameters to loss="perceptron", learning_rate = constant, 
#and eta = 0.1 and penalty = None.

##Multi-layer perceptron and backpropagation: 
#An MLP is composed of one input layer, one or more layers of LTUs,
#called hidden layers, and one final layer of LTUs called the output layer.
#Every layer except the output layer includes a bias neuron and is fully
#connected to the next layer. 

#Backpropagation explained: for each training instance the back propagation 
#algorithm first makes a prediction (forward pass), measures the error, then goes
#through each layer in reverse to measure the error contribution from each 
#connect (reverse pass), and finally slightly tweaks the connection weights 
#to reduce the error (Gradient Descent step). 
#The early activation function was replaced by the sigmoid equation 
#(1/1 + exp(-z)) as a means to make gradient descent work with the 
#model. 

##Important reason why the author in Fundamentals of deep learning used 
#softmax logistic regression: When classes are exclusive, the output layer 
#is typically modified by replacing the individual activation functions by 
#a shared soft max function. The output of each neuron corresponds to the estimated 
#probability of the corresponding class. 

##Training a MLP with Tensroflow's High level API: 
#The DNNClassifier class makes it fairly easy to train a deep neural 
#network with any number of hidden layers, and a softmax output layer 
#to output extimated class probabilities. 

import tensorflow as tf 

#(X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data() 
#X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
#X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
#y_train = y_train.astype(np.int32) 
#y_test = y_test.astype(np.int32) 
#X_valid, X_train = X_train[:5000], X_train[5000:] 
#y_valid, y_train = y_train[:5000], y_train[5000:] 

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original") 
X, y = mnist["data"], mnist["target"] 

X_train, X_test = X[:50000], X[50000:]
y_train, y_test = y[:50000], y[50000:] 
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train = y_train.astype(np.int32) 
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:] 
y_valid, y_train = y_train[:5000], y_train[5000:]   

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]  
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = [300, 100], n_classes = 10, 
	feature_columns= feature_cols) 
input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"X": X_train}, y=y_train, num_epochs = 40, batch_size=50, shuffle=True)
#dnn_clf.train(input_fn=input_fn) 
#Switched the mnist dataset from the sklearn.datasets directory and it works just fine.
#Now the problem is that the dnn_clf.train() function doesn't seem to want to work.
#Will need to read the documentation about this. 
tf.reset_default_graph() 

#Under the hood, the DNNClassifier class creates all the neuron layers, based on 
#the ReLu acitivation function (we can change this by setting the activation_fn hyperparameter).
#The output layer relies on the softmax function, and the cost function is 
#cross entropy. 

##Training a DNN Using Plain Tensorflow:
##construction phase: 
n_inputs = 28 * 28 #The total number of pixels beinging inputed into 
#the model. In other words, one pixel per feature.  
n_hidden1 = 300 
n_hidden2 = 100 
n_outputs = 10 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")#What about the bias 
#term? 
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
#this neural network is using the same code as the ReLU activation 
#neural network on page 247. thus meaning that this model is using 
#the relu activation function to create the neural network. 

#the stddev = 2 / np.sqrt(n_inputs) init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
#parts of the function is a Gaussian initialization number generator with a 
#standard deviation of 2/sqrt(n_inputs) 

#the first hidden layer takes X as its input. The second takes the output 
#of the first hidden layer as its input. And finally, the output layer takes 
#the output of the second hidden layer as its input. 

with tf.name_scope("dnn"):
	hidden1 = neuron_layer(X, n_hidden1, name = "hidden1",
		activation=tf.nn.relu) 
	hidden2 = neuron_layer(hidden1, n_hidden2, name = "hidden2",
		activation=tf.nn.relu) 
	logits = neuron_layer(hidden2, n_outputs, name = "outputs")

#Tensorflow's tf.layers.dense() function creates a fully connected layer, where 
#all the inputs are connected to al the neurons in the layer. It takes care of 
#creating the weights and biases variables, named kernel and bias respectively 
#using the appropriate initialization strategy, and you can set the 
#activation function using the activation argument. 

#with tf.name_scope("dnn"):
	#hidden1 = neuron_layer(X, n_hidden1, name = "hidden1",
		#activation=tf.nn.relu) 
	#hidden2 = neuron_layer(hidden1, n_hidden2, name = "hidden2",
		#activation=tf.nn.relu) 
	#logits = neuron_layer(hidden2, n_outputs, name = "outputs") 
#As you can see this creates the same neural network without having to 
#create a neuron function that specifies the variables within each 
#neuron in a specific layer.  

#After this we will need to assign a penality term within the equation.
#this this model we will use cross entropy. We will use sparse_soft_max_entropy_with_logits():
#it computes the cross entropy based on the logits. We can use tensorflow's 
#reduce_mean() to compute the mean cross entropy over all instances. 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

#the following steps are just like the ones used in chapter 9 to create 
#the linear regression computational map. 

learning_rate = 0.01 

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	training_op = optimizer.minimize(loss) 

#The last important step in the construction phase is to specify how to evaluate 
#the model. We will simply use accuracy as our performance measure. First, for each 
#instance, determine if the neural network's prediction is correct by checking whether
#or not the highest logit corresponds to the target class. For this you can use 
#the in_top_k() function. this returns a 1 D tensor full of boolean values, so we need 
#to cast these booleans to floats and then computate the average. this will give us 
#the network's overall accuracy.

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

n_epochs = 60 
batch_size = 50
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid) 
#Running the testing dataset, I can't get to an accuracy rating of over 62 percent 
#even with an n_epoch argument of 100. This most likely means that the indexes between the 
#testing set and training set weren't properly shuffled. 








