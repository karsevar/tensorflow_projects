###Hands on machine learning with Scikit learning and tensorflow
###chapter 11 Training Deep Neural Nets:

##Vanishing and Exploding Gradients problem:
##In this section the author mainly talks about the deep neural network problem 
#of exploding gradients (that plagues the upper layers of extremely large neural networks) 
#and the vanishing gradients problem. To learn more about this and the remedies 
#look at pages 277 and 278. 

#Popular fixes includes the Xavier initialization technique and the He 
#initialization technique. 

#By default, the tf.layers_dense() function uses Xavier initialization 
# (with a uniform distribution). You can change this to He initialization 
#by using the variance_scaling_initializer() function:
import tensorflow as tf 

X = tf.placeholder(tf.float32, shape =(None, 28*28), name = "X") 
n_hidden1 = 100 
#This is just to force these lines of code to not return an error after 
#running the he_init and hidden1 commands. 

he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, 
	kernel_initializer=he_init, name = "hidden1") 

tf.reset_default_graph()

##Nonsaturating Acivation functions:
#The ReLU activation function suffers from a problem known as the dying ReLUs: during training 
#some neurons effectively die, meaning they stop outputting anything other than 
#zero. 

#Fixes for this problem are leaky ReLUs which solves the dying problem through the hyperparameter 
#sigma (which controls how much the ReLU activation function will leak), randomized leaky ReLU, 
#parametric leaky ReLU, and lastly the Exponential linear unit (which is the newest and best received). 

#Tensorflow offers an elu() function that you can use to build your neural network. simply set the activation 
#argument when calling the dense() function, like this:

#hidden1 = tf.layers(X, n_hidden1, activation=tf.nn.elu, name = "hidden1") 

#tensorflow does not have a predefined function for leaky ReLUs, but it is easy to 
#define.
#def leaky_relu(z, name = None):
	#return tf.maximum(0.01 * z, z, name = name) 

#hidden1 = tf.layers.dense(X, n_hidden1, activation = leaky_relu, name = "hidden1") 

#It seems that all of these commands are broken. Will need to see if this is the case when trying to 
#use them in a computational graph. From what I can see is that the tf.nn.elu might be uncallable. 
#Will need to find an alternative to the elu function given by the author. 

##Batch Normalization:
#Another solution to the vanishing gradient problem: 
#the technique consists of adding an operation in the model just before the activaiton function of each layer, simply 
#zero-centering and normalizing the inputs, then scaling and shifting the result using two new parameters per layer 
#(one for scaling, the other for shifting). In other words, this operation lets the model 
#learn the optimal scale and mean of the inputs for each layer.

#In order to zero-center and normalize the inputs, the algorithm needs to esimate 
#the inputs' mean and standard deviation. It does so by evaluating the mean and standard 
#deviation of the inputs over the current mini batch. Look at page 282 for more information.

##Implementing Batch Normalization with tensorflow:
#Using the tf.layers.batch_normalization() function

n_inputs = 28 * 28 
n_hidden1 = 300 
n_hidden2 = 100 
n_outputs = 10 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X") 

training = tf.placeholder_with_default(False, shape=(), name = "training") 

hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1") 
bn1 = tf.layers.batch_normalization(hidden1, training = training, momentum = 0.9) 
bn1_act = tf.nn.elu(bn1) 
hidden2 = tf.layers.dense(X, n_hidden2, name = "hidden2") 
bn2 = tf.layers.batch_normalization(hidden2, training = training, momentum = 0.9) 
bn2_act = tf.nn.elu(bn2) 
logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = "outputs") 
logits = tf.layers.batch_normalization(logits_before_bn, training = training, momentum = 0.9) 
tf.reset_default_graph()
#Cool false alarm on the preceeding lines of code, the tf.nn.elu() activation function works perfectly.

#this code can be modified using the functools partial function (using the mnist dataset to create 
#a normalized batch neural network):
from functools import partial 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_mldata 

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"] 

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=20000, random_state = 42, shuffle = True)
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train = y_train.astype(np.int32) 
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:] 
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 100 
n_outputs = 10 
learning_rate = 0.01 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")

with tf.name_scope("dnn"):

	training = tf.placeholder_with_default(False, shape=(), name = "training")

	my_batch_norm_layer = partial(tf.layers.batch_normalization, 
		training = training, momentum = 0.9)

	hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1") 
	bn1 = tf.layers.batch_normalization(hidden1) 
	bn1_act = tf.nn.elu(bn1) 
	hidden2 = tf.layers.dense(bn1_act, n_hidden2, name = "hidden2") 
	bn2 = tf.layers.batch_normalization(hidden2) 
	bn2_act = tf.nn.elu(bn2) 
	logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = "outputs") 
	logits = tf.layers.batch_normalization(logits_before_bn)
#The partial function is a wrapper that contains the arguments training = training and 
#momentum = 0.9 so that you don't have to type them into every call the 
#batch_normalization.

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	training_op = optimizer.minimize(loss)


def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

n_epochs = 20 
batch_size = 200
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
		accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Validation accuracy:", accuracy_val) 

	save_path = saver.save(sess, "./my_model_final2.ckpt")

tf.reset_default_graph() 
#this method converged at 0.932 accuracy rate with the validation set which is 
#ok when one looks at the fact that it achieved this number in only 20 iterations.
#A better accuracy rating can be achieved through more iterations and a larger neural network
#(more hidden layers with more neurons). 

##Gradient clipping: 
#A popular technique to lessen the exploding gradients problem is to simply clip 
#the gradients during backpropagation so that they never exceed some threshold.

#In tensorflow, the optimizer's minimize() function takes care of both computing the 
#gradients and applying them, so you must instead call the optimizer compute_gradients()
#method first , then create an operation to clip the gradients using the clip_by_value 
#function, and finally create an operation to apply the clipped gradients using the optimizer's 
#apply_gradients() method: 

#Gradient clipping implementation using the ReLU activation function and the mnist 
#dataset.

n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 100 
n_outputs = 10 
learning_rate = 0.01
n_epochs = 10  
batch_size = 50
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size)) 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

def neuron_layer(X, n_neurons, name, activation = None):
	with tf.name_scope("ReLU"):
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

with tf.name_scope("train"):
	threshold = 1.0 
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	grads_and_vars = optimizer.compute_gradients(loss) 
	capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
					for grad, var in grads_and_vars]
	training_op = optimizer.apply_gradients(capped_gvs) 

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

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

tf.reset_default_graph()
#Sweet this implementation works perfectly!!! 

##Reusing Pretrained layers:
#transfer learning: the process of reusing lower leval neural network layers to 
#solve a different machine learning task. It is important to keep in mind that this 
#method only works if the two networks have the same sizes (thus before transfering 
#ones data to the older neural network, be sure to resize the input data).

##Reusing Tensorflow Model: 
#You can use this function to import the operations into 
#the default graph. This returns a Saver that you can later use to load the 
#model's state.

saver = tf.train.import_meta_graph("./my_model_final2.ckpt.meta") 
#Next you need to find the names of the operations within the saved tensorflow session.

for op in tf.get_default_graph().get_operations():
	print(op.name) 

X = tf.get_default_graph().get_tensor_by_name("X:0") 
y = tf.get_default_graph().get_tensor_by_name("y:0") 
accuracy = tf.get_default_graph().get_tensor_by_name("eval/Mean:0") 
training_op = tf.get_default_graph().get_operation_by_name("train/GradientDescent")
#Finally found the labels within my batch normalization graph. I hope these are the correct 
#modules. 
with tf.Session() as sess:
	saver.restore(sess, "./my_model_final2.ckpt") 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid)
	
	save_path = saver.save(sess, "./my_new_model_final2.ckpt")
tf.reset_default_graph()

#splicing on new hidden layers onto existing neural network frameworks:
n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 100 
n_outputs = 10 
learning_rate = 0.01
n_epochs = 10  
batch_size = 100 
threshold = 1.0
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size)) 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2")
	logits = tf.layers.dense(hidden2, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
grads_and_vars = optimizer.compute_gradients(loss) 
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
				for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs) 

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run() 

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid)
	
	save_path = saver.save(sess, "./my_new_model_final3.ckpt")

tf.reset_default_graph()

#Neural network splicing experiment:
n_hidden3 = 100 
n_outputs = 10 

saver = tf.train.import_meta_graph("./my_new_model_final3.ckpt.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0") 
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden2 = tf.get_default_graph().get_tensor_by_name("dnn/hidden2/Relu:0")
new_hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.relu, name = "new_hidden3") 
new_logits = tf.layers.dense(new_hidden3, n_outputs, name = "new_outputs") 

with tf.name_scope("new_loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = new_logits) 
	loss = tf.reduce_mean(xentropy, name = "loss") 

with tf.name_scope("new_eval"):
	correct = tf.nn.in_top_k(new_logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("new_train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer() 
new_saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run() 
	saver.restore(sess, "./my_new_model_final3.ckpt")

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid)
	
	save_path = new_saver.save(sess, "./my_new_model_final4.ckpt") 

tf.reset_default_graph() 
##Freezing the Lower Layers:
#It is generally a good idea to freeze the lower layers weights during training: If the lower 
#layer weights are fixed, then the higher layer weights will be easier to train (because they won't 
#have to learn a moving target). 

#The layer freezing technique using tf.stop_gradient() 
n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 50 
n_hidden3 = 50 
n_hidden4 = 20
n_outputs = 10 
learning_rate = 0.01
n_epochs = 10  
batch_size = 100 
threshold = 1.0
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size)) 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2")
	hidden2_stop = tf.stop_gradient(hidden2) #This line stops the training process within hidden layer 
	#2. This technique is very important with reusing lower layers within different machine learning problems as 
	#well as descrease the time needed to train a model.
	hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.relu, name = "hidden3")
	hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu, name = "hidden4") 
	logits = tf.layers.dense(hidden4, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	training_op = optimizer.minimize(loss) 

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
	scope = "hidden[123]")
restore_saver = tf.train.Saver(reuse_vars)  

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run() 

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
		print(epoch, "Train accuracy: ", acc_train, "valid accuracy:", acc_valid)
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()# frozen lower layer code works perfectly.

##Caching the frozen Layers:
import numpy as np 

n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 50 
n_hidden3 = 50 
n_hidden4 = 20
n_outputs = 10 
learning_rate = 0.01
n_epochs = 10  
batch_size = 100 
threshold = 1.0
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2")
	hidden2_stop = tf.stop_gradient(hidden2) #This line stops the training process within hidden layer 
	#2. This technique is very important with reusing lower layers within different machine learning problems as 
	#well as descrease the time needed to train a model.
	hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.relu, name = "hidden3")
	hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu, name = "hidden4") 
	logits = tf.layers.dense(hidden4, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	training_op = optimizer.minimize(loss) 

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
	scope = "hidden[123]")
restore_saver = tf.train.Saver(reuse_vars)  

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	restore_saver.restore(sess, "./my_model_final.ckpt")

	h2_cache = sess.run(hidden2, feed_dict = {X:X_train}) 
	h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid})

	for epoch in range(n_epochs):
		shuffle_idx = np.random.permutation(len(X_train))
		hidden2_batches = np.array_split(h2_cache[shuffle_idx], n_batches)
		y_batches = np.array_split(y_train[shuffle_idx], n_batches) 
		for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
			sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})
		accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_valid, 
												y: y_valid})

		print(epoch, "Validation accuracy:", accuracy_val)
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()

##Faster optimizers:
#The most popular ones are: Momentum optimization, Nesterov Accelerated 
#Gradient, AdaGrad, RMSProp, and finally Adam optimization.

##Momentum optimization:
#Momentum gradient descent cares a great deal about what previous gradients were: at 
#each iteration, it adds the local gradient to the momentum vector m (multiplied by the learning rate) 
#, and it updates the weights by simply subtracting this momentum vector. To simulate some sort of 
#friction, the algorithm introduces a new hyperparameter beta, simply called momentum and it's on 
#a 0 (high friction) to 1 (no friction) scale. 

#implementation: 
import numpy as np 

n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 50 
n_hidden3 = 50 
n_hidden4 = 20
n_outputs = 10 
learning_rate = 0.01
n_epochs = 10  
batch_size = 100 
threshold = 1.0
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2")
	hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.relu, name = "hidden3")
	hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu, name = "hidden4") 
	logits = tf.layers.dense(hidden4, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
	#This transforms the gradient descent optimizer into a momentum gradient descent optimizer. 
	training_op = optimizer.minimize(loss)   

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

		print(epoch, "Validation accuracy:", acc_valid) 
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()

##Neserov Accelerated Gradient:
#Just like momentum gradient except that the formula's author measures the 
#gradient with the term theta + beta*m rather than just theta with the momentum 
#method. This increases the speed of convergence and keeps the algorithm from over 
#shotting the global minimum. 

#Implementation:
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2")
	hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.relu, name = "hidden3")
	hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu, name = "hidden4") 
	logits = tf.layers.dense(hidden4, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9, use_nesterov = True) 
	#Change the use_nesterov argument to True. 
	training_op = optimizer.minimize(loss)   

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

		print(epoch, "Validation accuracy:", acc_valid) 
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()

##AdaGrad:
#perfect for linear regression gradient descent tasks as it corrects for the bowl 
#shape of the cost function (convex function) but it is ineffective for neural networks 
#for that same reason. 
#Look at page 299 for more information.

##RMSProp:
#this algorithm has the same characteristics as the AdaGrad algorithm with the main 
#difference being the hyperparameter beta and accumulating the most recent iterations of the 
#gradient. 

#this algorithm can be implemented with the tf.train.RMSPropOptimizer() function. 

##Adam Optimizer: 
#Look at page 300 to see the full description. This model is more or less a high bred between the 
#models listed above. 

#Implementation is achieved through the tf.train.AdamOptimizer() function. 

##Learning Rate Scheduling:
#The methods are: Predetermined piecewise constant learning rate, Performance scheduling, 
#Exponential scheduling, and Power scheduling.

#Implementing a learning schedule in tensorflow:

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")  

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2")
	hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.relu, name = "hidden3")
	hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.relu, name = "hidden4") 
	logits = tf.layers.dense(hidden4, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	initial_learning_rate = 0.1 
	decay_steps = 10000
	decay_rate = 1/10 
	global_step = tf.Variable(0, trainable=False, name = "global_step")
	learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 
										decay_steps, decay_rate)
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
	training_op = optimizer.minimize(loss, global_step = global_step)      

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

		print(epoch, "Validation accuracy:", acc_valid) 
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()

##Avoiding overfitting through regularization:
##Regulatization:
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")

learning_rate = 0.01 
scale = 0.001

my_dense_layer = partial(
	tf.layers.dense, activation = tf.nn.relu, 
	kernel_regularizer = tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("dnn"):
	hidden1 = my_dense_layer(X, n_hidden1, name = "hidden1") 
	hidden2 = my_dense_layer(hidden1, n_hidden2, name = "hidden2") 
	logits = my_dense_layer(hidden2, n_outputs, activation = None, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	base_loss = tf.reduce_mean(xentropy, name = "avg_xentropy")
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss = tf.add_n([base_loss] + reg_losses, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)      

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

		print(epoch, "Validation accuracy:", acc_valid) 
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()

##Drop out method implementation:
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")
training = tf.placeholder_with_default(False, shape = (), name = "training") 

dropout_rate = 0.5 
X_drop = tf.layers.dropout(X, dropout_rate, training = training) 

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X_drop, n_hidden1, activation = tf.nn.relu, 
						name = "hidden1") 
	hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training = training) 
	hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation = tf.nn.relu, name = "hidden2") 
	hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training = training)
	logits = tf.layers.dense(hidden2_drop, n_outputs, activation = None, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9)
	training_op = optimizer.minimize(loss)      

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

		print(epoch, "Validation accuracy:", acc_valid) 
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()

##Max norm implementation:
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")

n_inputs = 28 * 28 
n_hidden1 = 300
n_hidden2 = 50 
n_outputs = 10 

learning_rate = 0.01 
momentum = 0.9 

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.relu, name = "hidden2") 
	logits = tf.layers.dense(hidden2, n_outputs, activation = None, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss) 

threshold = 1.0 
weights = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
clipped_weights = tf.clip_by_norm(weights,clip_norm = threshold, axes = 1)

weights2 = tf.get_default_graph().get_tensor_by_name("hidden2/kernel:0") 
clipped_weights2 = tf.clip_by_norm(weights2, clip_norm=threshold, axes = 1) 
clip_weights2 = tf.assign(weights2, clipped_weights2)      

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			clip_weights.eval() 
			clip_weights2.eval()
		acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

		print(epoch, "Validation accuracy:", acc_valid) 
	
	save_path = saver.save(sess, "./my_model_final.ckpt")

tf.reset_default_graph()




