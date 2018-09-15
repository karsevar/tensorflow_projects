###Hands on machine learning with scikit learn and tensorflow.
##Chapter 15 Autoencoders: 

#Autoencoders are artificial neural networks capable of learning efficient 
#representations of the input data, called codings, without any supervision.
#Autoencoders are powerful feature detectors, and they can be used for 
#supervised pretraining of deep neural networks. Lastly they are capable 
#of randomly generating new data that looks very similar to the training data;
#this is called a generative model.

##Efficient Data representation: 
#an autoencoder looks at the inputs, converts them to an efficient internal 
#representation, and then spits out something that looks very close to the inputs
#An autoencoder is always composed of two parts: an encoder (or recognition network) 
#that converts the inputs to an internal representation, followed by a decoder 
#(or generative network) that converts the internal representation 
#to the outputs. 

#typically autoencoder have the same architecture as a multi layer 
#perceptron, except that the number of neurons in the output layer 
#must be equal to the number of inputs. The outputs are often called the 
#reconstructions since the autoencoder tries to reconstruct the inputs,
#and the cost function contains a reconstruction loss that penalizes the 
#model when the reconstructions are different from the inputs. 

#Because the internal representation has a lower dimensionality than the input 
#data, the autoencoder is said to be incomplete. An incomplete autoencoder 
#cannot trivially copy its inputs to the codings, yet it must find a way to 
#output a copy of its inputs. It is forced to learn the most important 
#features in the input data (and drop the unimportant ones). 

##Performing PCA with an Undercomplete linear autoencoder. 
#If the autoencoder uses only linear activations and the cost function is 
#the mean squared error, then it can be shown that it ends up performing 
#Principal component analysis. 

import tensorflow as tf 

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs 
#the following code will most likely transform an input of size 3 dimensions 
#into an output of 2 dimensions. 

learning_rate = 0.01 

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 
hidden = tf.layers.dense(X, n_hidden) 
outputs = tf.layers.dense(hidden, n_outputs) 

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) #MSE 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
train_op = optimizer.minimize(reconstruction_loss) 

init = tf.global_variables_initializer()

#The couple things to note about this code: 
#	The number of outputs is equal to the number of inputs 
#	To perform simple PCA, we do not use any activation function and the 
#		cost function is the MSE. 

import numpy.random as rnd 
import numpy as np 

rnd.seed(4) 
m = 200 
w1, w2 = 0.1, 0.3 
noise = 0.1 

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3)) 
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:,2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m) 

#Normalize the data:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
X_train = scaler.fit_transform(data[:100]) 
X_test = scaler.transform(data[100:])
#One hundred observations for the training set and one hundred for the 
#testing set. Notice that the testing set is only transformed with the 
#scaler object as the training set dictates the fit of the normalization 
#scale. 

##Now let's run the model: 
n_iterations = 1000 
codings = hidden 

with tf.Session() as sess: 
	init.run() 
	for iteration in range(n_iterations):
		train_op.run(feed_dict={X: X_train})
		train_accur = reconstruction_loss.eval(feed_dict={X: X_train}) 
		print("Training mean squared error:", train_accur)  
	codings_val = codings.eval(feed_dict={X:X_test})
	test_accur = reconstruction_loss.eval(feed_dict={X: X_test}) 
	print("testing mean squared error:", test_accur) 

import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(4,3)) 
plt.plot(codings_val[:, 0], codings_val[:, 1], "b.") 
plt.xlabel("$z_1$", fontsize= 18) 
plt.ylabel("$z_2$", fontsize = 18, rotation = 0) 
#plt.show()

##Stacked Autoencoder: 
#Just like other neural networks we have discussed, autoencoders can have 
#multiple hidden layers. In this case they are called stacked autoencoders 
#(or deep autoencoders). Adding more layers helps the autoencoder learn 
#more complex codings. However, one must be careful not to make the autoencoder 
#too powerful (as the model won't be able to generalize the underlying 
#data into smaller groups). 

#The architecture of a stacked autoencoder is typically symmetrical with 
#regards to the central hidden layer (the coding layer). For example, an autoencoder 
#for MNIST may have 784 inputs, followed by a hidden layer with 300 neurons, then 
#a central hidden layer of 150 neurons, then another hidden layer with 300 
#neurons, and an output layer with 784 neurons. 

#The following code builds a stacked autoencoder for MNIST, using He 
#initialization, the ELU activation function, and l2 regularization.
#Remember that the labels y will be left out of this model. 

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split  

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, 
															random_state = 42, shuffle = True)

tf.reset_default_graph() 

from functools import partial 

n_inputs = 28 * 28 
n_hidden1 = 300 
n_hidden2 = 150 #codings 
n_hidden3 = 300 
n_outputs = 28 * 28 

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 

he_init = tf.contrib.layers.variance_scaling_initializer() 
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg) 
my_dense_layer = partial(tf.layers.dense, 
						activation=tf.nn.elu,
						kernel_initializer=he_init,
						kernel_regularizer=l2_regularizer) 

hidden1 = my_dense_layer(X, n_hidden1) 
hidden2 = my_dense_layer(hidden1, n_hidden2)  
hidden3 = my_dense_layer(hidden2, n_hidden3) 
outputs = my_dense_layer(hidden3, n_outputs, activation = None) 

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) 

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses) 

optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer() 

n_epochs = 5 
batch_size = 150 
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict = {X: X_batch}) 
		print(epoch)
	print("done training") 


##tying Weights:
#When an autoencoder is neatly symmetrical, like the one we just built, a common 
#technique is to tie the weights of the decoder layers to the weights of the encoder 
#layers. this halves the number of weights in the model, speeding up training and 
#limiting the risk of overfitting. 

#The best way to implement this technique is to write the neurons by 
#hand into a computation graph without the help of tf.layers.dense().

tf.reset_default_graph()

X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 

n_inputs = 28 * 28 
n_hidden1 = 300 
n_hidden2 = 150 
n_hidden3 = n_hidden1
n_outputs = n_inputs 

learning_rate = 0.01 
l2_reg = 0.0005

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer() #He initializer

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 

weights1_init = initializer([n_inputs, n_hidden1]) 
weights2_init = initializer([n_hidden1, n_hidden2]) 

weights1 = tf.Variable(weights1_init, dtype = tf.float32, name = "weights1") 
weights2 = tf.Variable(weights2_init, dtype = tf.float32, name = "weights2") 
weights3 = tf.transpose(weights2, name = "weights3") #tied weights 
weights4 = tf.transpose(weights1, name = "weights4") #tied weights 

biases1 = tf.Variable(tf.zeros(n_hidden1), name = "biases1") 
biases2 = tf.Variable(tf.zeros(n_hidden2), name = "biases2") 
biases3 = tf.Variable(tf.zeros(n_hidden3), name = "biases3") 
biases4 = tf.Variable(tf.zeros(n_outputs), name = "biases4") 

hidden1 = activation(tf.matmul(X, weights1) + biases1) 
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2) 
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3) 
outputs = tf.matmul(hidden3, weights4) + biases4 

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) 
reg_loss = regularizer(weights1) + regularizer(weights2) 
loss = reconstruction_loss + reg_loss 

optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer() 
saver = tf.train.Saver()

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			#print("\r{}%".format(100 * batch_index // n_batches), end = "")
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict = {X: X_batch}) 
		loss_train = reconstruction_loss.eval(feed_dict={X:X_batch})
		print("\r{}".format(epoch), "Train MSE:", loss_train) 
	print("done training")
	saver.save(sess, "./my_model_tying_weights.ckpt")  

#function lifted from the book's github: 
def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
	with tf.Session() as sess:
		if model_path:
			saver.restore(sess, model_path)
		outputs_val = outputs.eval(feed_dict = {X:X_test})

	fig = plt.figure(figsize = (8, 3 * n_test_digits))
	for digit_index in range(n_test_digits):
		plt.subplot(n_test_digits, 2, digit_index * 2 + 1) 
		plot_image(X_test[digit_index]) 
		plt.subplot(n_test_digits, 2, digit_index * 2 + 2) 
		plot_image(outputs_val[digit_index])

def plot_image(image, shape = [28, 28]):
	plt.imshow(image.reshape(shape), cmap = "Greys", interpolation = "nearest") 
	plt.axis("off") 

##Training one autoencoder at a time: 
tf.reset_default_graph() 

n_inputs = 28 * 28 
n_hidden1 = 300 
n_hidden2 = 150 
n_hidden3 = n_hidden1 
n_outputs = n_inputs 

learning_rate = 0.01 
l2_reg = 0.0001 

activation = tf.nn.elu 
regularizer = tf.contrib.layers.l1_regularizer(l2_reg) 
initializer = tf.contrib.layers.variance_scaling_initializer() 

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 

weights1_init = initializer([n_inputs, n_hidden1]) 
weights2_init = initializer([n_hidden1, n_hidden2]) 
weights3_init = initializer([n_hidden2, n_hidden3]) 
weights4_init = initializer([n_hidden3, n_outputs]) 

weights1 = tf.Variable(weights1_init, dtype = tf.float32, name = "weights1") 
weights2 = tf.Variable(weights2_init, dtype = tf.float32, name = "weights2") 
weights3 = tf.Variable(weights3_init, dtype = tf.float32, name = "weights3") #tied weights 
weights4 = tf.Variable(weights4_init, dtype = tf.float32, name = "weights4") #tied weights 

biases1 = tf.Variable(tf.zeros(n_hidden1), name = "biases1") 
biases2 = tf.Variable(tf.zeros(n_hidden2), name = "biases2") 
biases3 = tf.Variable(tf.zeros(n_hidden3), name = "biases3") 
biases4 = tf.Variable(tf.zeros(n_outputs), name = "biases4") 

hidden1 = activation(tf.matmul(X, weights1) + biases1) 
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2) 
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3) 
outputs = tf.matmul(hidden3, weights4) + biases4 

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  
optimizer = tf.train.AdamOptimizer(learning_rate)

with tf.name_scope("phase1"):
	phase1_outputs = tf.matmul(hidden1, weights4) + biases4 #bypass hidden2 and hidden3 
	phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
	phase1_reg_loss = regularizer(weights1) + regularizer(weights4) 
	phase1_loss = phase1_reconstruction_loss + phase1_reg_loss 
	phase1_training_op = optimizer.minimize(phase1_loss) 

with tf.name_scope("phase2"):
	phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
	phase2_reg_loss = regularizer(weights2) + regularizer(weights3) 
	phase2_loss = phase2_reconstruction_loss + phase2_reg_loss 
	train_vars = [weights2, biases2, weights3, biases3] 
	phase2_training_op = optimizer.minimize(phase2_loss, var_list = train_vars) # freeze hidden1

init = tf.global_variables_initializer() 
saver = tf.train.Saver()

training_ops = [phase1_training_op, phase2_training_op] 
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss] 
n_epochs = [4, 4]
batch_sizes = [150, 150] 

with tf.Session() as sess: 
	init.run() 
	for phase in range(2):
		print("Training phase #{}".format(phase + 1))
		for epoch in range(n_epochs[phase]):
			for batch_index in range(n_batches):
				X_batch, y_batch = fetch_batch(epoch, batch_index, batch_sizes[phase]) 
				sess.run(training_ops[phase], feed_dict = {X: X_batch}) 
			loss_train = reconstruction_losses[phase].eval(feed_dict={X:X_batch})
			print("\r{}".format(epoch), "Train MSE:", loss_train) 
			saver.save(sess, "./my_model_one_at_a_time.ckpt") 
	loss_test = reconstruction_loss.eval(feed_dict = {X: X_test})
	print("Test MSE:", loss_test)

#Caching the frozen layer outputs:
training_ops = [phase1_training_op, phase2_training_op] 
reconstructrion_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss] 
n_epochs = [4, 4] 
batch_sizes = [150, 150]
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size)) 

with tf.Session() as sess: 
	init.run() 
	for phase in range(2):
		print("Training phase #{}".format(phase + 1))
		if phase == 1:
			hidden1_cache = hidden1.eval(feed_dict={X:X_train}) 
		for epoch in range(n_epochs[phase]):
			for batch_index in range(n_batches):
				print("\r{}%".format(100 * batch_index // n_batches), end = "") 
				if phase == 1:
					indices = rnd.permutation(len(X_train))
					hidden1_batch = hidden1_cache[indices[:batch_sizes[phase]]]
					feed_dict = {hidden1: hidden1_batch}
					sess.run(training_ops[phase], feed_dict = feed_dict) 
				else:
					X_batch, y_batch = fetch_batch(epoch, batch_index, batch_sizes[phase])
					feed_dict = {X: X_batch}
					sess.run(training_ops[phase], feed_dict =feed_dict) 
			loss_train = reconstruction_losses[phase].eval(feed_dict=feed_dict)
			print("\r{}".format(epoch), "Train MSE:", loss_train) 
			saver.save(sess, "./my_model_cache_frozen.ckpt") 
	loss_test = reconstruction_loss.eval(feed_dict = {X: X_test})
	print("Test MSE:", loss_test)


##Visualizing the Reconstructions: 
#One way to ensure that an autoencoder is properly trained is to compare 
#the inputs and the outputs. They must be fairly similar, and the differences 
#should be unimportant details. 

n_test_digits = 2
X_test_image = X_test[:n_test_digits] 

with tf.Session() as sess: 
	saver.restore(sess, "./my_model_one_at_a_time.ckpt") 
	outputs_val = outputs.eval(feed_dict={X:X_test_image}) 

def plot_image(image, shape=[28,28]):
	plt.imshow(image.reshape(shape), cmap = "Greys", interpolation = "nearest") 

for digit_index in range(n_test_digits):
	plt.subplot(n_test_digits, 2, digit_index * 2 + 1) 
	plot_image(X_test_image[digit_index]) 
	plt.subplot(n_test_digits, 2, digit_index * 2 + 2) 
	plot_image(outputs_val[digit_index]) 
	plt.show() 
	#Not a bad representation but still I can see that an 
	#algorithm might misclassify the numbers. The features used 
	#might be incapable of creating a model that is capable of 
	#classifying the hand written number correctly. 

##Visualizing Features: 
#Once your autoencoder has learned some features, you may want to take 
#a look at the. The simplest technique to look at the features created by 
#the encoder is to consider each neuron in every hidden layer, and find 
#the training instances that activate it the most. this is especially 
#useful for the top hidden layers since they often capture relatively 
#large features that you can easily spot in a group of training instances.

#Let's look at another technique. for each neuron in the first 
#hidden layer, you can create an image where a pixel's intensity 
#corresponds to the weight of the connection to the given neuron.
#For example, the following code plots the features learned by five neurons 
#in the first hidden layer. 

with tf.Session() as sess: 
	saver.restore(sess, "./my_model_one_at_a_time.ckpt") 
	weights1_val = weights1.eval() 

for i in range(5):
	plt.subplot(1, 5, i + 1) 
	plot_image(weights1_val.T[i]) 
	plt.show() 

#Finally, if you are using an autoencoder to perform unsupervised 
#pretraining -- for example, for a classification task -- a simple way 
#to verify that the features learned by the autoencoder are useful is to measure 
#the performance of the classifier. 

##Unsupervised Pretraining Using Stacked autoencoders:
#If you have a large dataset but most of it is unlabeled, you can 
#first train a stacked autoencoder using all the data, then reuse the lower 
#layers to create a neural network for your actual task, and train it using 
#the labeled data. 

tf.reset_default_graph() 

n_inputs = 28 * 28 
n_hidden1 = 300 
n_hidden2 = 150 
n_outputs = 10 

learning_rate = 0.01 
l2_reg = 0.0005 

activation = tf.nn.elu 
regularizer = tf.contrib.layers.l2_regularizer(l2_reg) 
initializer = tf.contrib.layers.variance_scaling_initializer() 

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 
y = tf.placeholder(tf.int32, shape = [None]) 

weights1_init = initializer([n_inputs, n_hidden1]) 
weights2_init = initializer([n_hidden1, n_hidden2]) 
weights3_init = initializer([n_hidden2, n_outputs])  

weights1 = tf.Variable(weights1_init, dtype = tf.float32, name = "weights1") 
weights2 = tf.Variable(weights2_init, dtype = tf.float32, name = "weights2") 
weights3 = tf.Variable(weights3_init, dtype = tf.float32, name = "weights3") 

biases1 = tf.Variable(tf.zeros(n_hidden1), name = "biases1") 
biases2 = tf.Variable(tf.zeros(n_hidden2), name = "biases2") 
biases3 = tf.Variable(tf.zeros(n_outputs), name = "biases3")  

hidden1 = activation(tf.matmul(X, weights1) + biases1) 
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2) 
logits = tf.matmul(hidden2, weights3) + biases3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3) 
loss = cross_entropy + reg_loss 
optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(loss) 

correct = tf.nn.in_top_k(logits, y, 1) 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 

init = tf.global_variables_initializer() 
pretrain_saver = tf.train.Saver([weights1, weights2, biases1, biases2]) 
saver = tf.train.Saver() 

n_epochs = 4 
batch_size = 150 
n_labeled_instances = 20000 

with tf.Session() as sess:
	init.run() 
	pretrain_saver.restore(sess, "./my_model_cache_frozen.ckpt") 
	for epoch in range(n_epochs):
		for iteration in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, iteration, batch_size) 
			sess.run(training_op, feed_dict = {X: X_batch, y: y_batch}) 
		accuracy_val = accuracy.eval(feed_dict = {X: X_batch, y: y_batch}) 
		print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end = "\t") 
		saver.save(sess, "./my_model_supervised_pretrained.ckpt") 
		accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test}) 
		print("Test Accuracy:", accuracy_val) 
#This model is a lot faster than the previous models but the only problem 
#I can see is that the accuracy value is unable to go above 95 percent for the 
#testing set over the course of 5 iterations. But still the increased error 
#rate is minor compared to how fast the model has able to be trained.

##Stacked denoising Autoencoder: 
tf.reset_default_graph() 

n_inputs = 28 * 28 
n_hidden1 = 300 
n_hidden2 = 150 
n_hidden3 = n_hidden1
n_outputs = n_inputs 

learning_rate = 0.01 
noise_level = 1.0 

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 
X_noisy = X + noise_level * tf.random_normal(tf.shape(X)) 

hidden1 = tf.layers.dense(X_noisy, n_hidden1, activation = tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu) 
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu) 
outputs = tf.layers.dense(hidden3, n_outputs) 
reconstruction_loss = tf.reduce_mean(tf.square(outputs-X)) 

optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(reconstruction_loss) 

init = tf.global_variables_initializer() 
saver = tf.train.Saver() 

n_epochs = 10 
batch_size = 150 

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epochs):
		for iteration in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, iteration, batch_size) 
			sess.run(training_op, feed_dict = {X:X_batch}) 
		loss_train = reconstruction_loss.eval(feed_dict={X:X_batch}) 
		print("\r{}".format(epoch), "Train MSE:", loss_train) 
		saver.save(sess, "./my_model_stacked_denoising_gaussian.ckpt") 

##Sparse Autoencoders: 
#to see the description of this model and why it is the most used 
#denoising autoencoding method look at page 430. 

tf.reset_default_graph() 

n_inputs = 28 * 28 
n_hidden1 = 1000 
n_outputs = n_inputs 

def k1_divergence(p, q):
	return p * tf.log(p/q) + (1-p) * tf.log((1 - p) / (1 - q))

learning_rate = 0.01 
sparsity_target = 0.1 
sparsity_weight = 0.2 

X = tf.placeholder(tf.float32, shape = [None, n_inputs]) 

hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid) 
outputs = tf.layers.dense(hidden1, n_outputs) 

hidden1_mean = tf.reduce_mean(hidden1, axis = 0) 
sparsity_loss = tf.reduce_sum(k1_divergence(sparsity_target, hidden1_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
loss = reconstruction_loss + sparsity_weight * sparsity_loss 

optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer() 
saver = tf.train.Saver() 

n_epochs = 5 
batch_size = 1000
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size)) 

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for iteration in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, iteration, batch_size) 
			sess.run(training_op, feed_dict = {X: X_batch}) 
		reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss,
			sparsity_loss, loss], feed_dict={X:X_batch}) 
		print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val, "\tSparsity loss:", 
			sparsity_loss_val, "\tTotal loss:", loss_val) 
		saver.save(sess, "./my_model_sparse.ckpt")

##Variational Autoencoders: 
#The following builds a variation autoencoder with a total of five hidden 
#layers. 

tf.reset_default_graph() 

from functools import partial 

n_inputs = 28 * 28 
n_hidden1 = 500 
n_hidden2 = 500 
n_hidden3 = 20 #codings 
n_hidden4 = n_hidden2 
n_hidden5 = n_hidden1 
n_outputs = n_inputs 
learning_rate = 0.001
n_digits = 60 

initializer = tf.contrib.layers.variance_scaling_initializer() 
my_dense_layer = partial( 
	tf.layers.dense, 
	activation=tf.nn.elu,
	kernel_initializer=initializer) 

X = tf.placeholder(tf.float32, [None, n_inputs]) 
hidden1 = my_dense_layer(X, n_hidden1) 
hidden2 = my_dense_layer(hidden1, n_hidden2) 
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None) 
hidden3_sigma = my_dense_layer(hidden2, n_hidden3, activation = None) 
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype = tf.float32) 
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5) 
logits = my_dense_layer(hidden5, n_outputs, activation = None) 
outputs = tf.sigmoid(logits) 

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = X, logits = logits) 
reconstruction_loss = tf.reduce_sum(xentropy) 

eps = 1e-10 
latent_loss = 0.5 * tf.reduce_sum(
	tf.square(hidden3_sigma) + tf.square(hidden3_mean) 
	- 1 - tf.log(eps + tf.square(hidden3_sigma)))

loss = reconstruction_loss + latent_loss 

optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

n_epochs = 50 
batch_size = 150 

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for iteration in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, iteration, batch_size) 
			sess.run(training_op, feed_dict = {X: X_batch}) 
		loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X:X_batch}) 
		print("\r{}".format(epoch), "Total loss:", loss_val, "\tReconstruction loss:", 
			reconstruction_loss_val, "\tLatent Loss:", latent_loss_val) 
		saver.save(sess, "./my_model_variational.ckpt")
	codings_rnd = np.random.normal(size=[n_digits, n_hidden3]) 
	outputs_val = outputs.eval(feed_dict={hidden3:codings_rnd})

plt.figure(figsize=(8,50)) 
for iteration in range(n_digits):
	plt.subplot(n_digits, 10, iteration + 1) 
	plot_image(outputs_val[iteration]) 

plt.show()  
























