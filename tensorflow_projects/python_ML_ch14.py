###Hands on machine learning with Scikit learning and tensorflow
###chapter 14 Recurrent Neural Networks:

##Recurrent Neurons:
#Up to now we have mostly looked at feedforward neural networks, where the activations 
#flow only in one direction, from the input layer to the output layer. A recurrent neural 
#network looks very much like the feedfoward neural network, except for 
#it also has connections pointing backwards. Let's look at the simplest possible 
#RNN, composed of just one neuron recieving inputs, producing an output, and sending 
#that output back to itself. 

#to see the computation of one recurrent neuron look at page 383.

##Memory cells: 
#Since the output of a recurrent neuron at time step t is a function of all the inputs
#from the previous time steps, you could say it has a form of memory. 
#A part of a neural network that preserves some state across time steps is 
#called a memory cell.

##Input and output sequences:
#An RNN can simultaneously take a sequence and produce da sequence of outputs.
#For example, this type of network is useful for predicting time series. You can feed 
#it prices over the last N days and it must output the prices shifted by one day into the 
#future. Alternatively, you could feed the network a sequence of inputs, and ignore 
#all the outputs except for the last one. In other words, this is a sequence to vector 
#network. 

#Conversely, you could feed the network a single input at the first time step, and 
#let it output a sequence. This is called a vector to sequence network. 

#Lastly you could have a sequence to vector network called an encoder, followed by a 
#vector to sequence network called a decoder. This is used for language to language translations 

##Basic RNNs in tensorflow: 
#We will create an RNN composed of a layer of five recurrent neurons using the tanh activation 
#function. We will assume that the RNN runs over only two time steps, taking input vectors of size 
#3 at each time step. 

import tensorflow as tf 

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs]) 
X1 = tf.placeholder(tf.float32, [None, n_inputs]) 

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype = tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype = tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype = tf.float32)) 

Y0 = tf.tanh(tf.matmul(X0, Wx) + b) 
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b) 

init = tf.global_variables_initializer() 

#This network looks much like a two-layer feedforward neural network, with a few twists
#first, the same weights and bias terms are shared by both layers, and second, we 
#feed inputs at each layer, and we get outputs from each layer. To run the model, we 
#need to feed it the inputs at both time steps, like so: 

import numpy as np 

X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]]) # t = 0 
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]]) # t = 1 

with tf.Session() as sess: 
	init.run() 
	Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})
	print(Y0_val, Y1_val) 

#Sweet this small RNN actually works!!!

#This mini batch contains four instances, each with an input sequence composed of 
#exactly two inputs. At the end, Y0_val and Y1_val contain the outputs of the network 
#at both time steps for all neurons and all instances in the mini_batch. 

##Static Unrolling through time: 
#The static_rnn() function creates an unrolling RNN network by chaining cells. The
#following code creates the exact same model as the one above. 

#Using the static_rnn() function 
tf.reset_default_graph() 

n_inputs = 3 
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs]) 
X1 = tf.placeholder(tf.float32, [None, n_inputs]) 

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) 
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype = tf.float32)

Y0, Y1 = output_seqs

init = tf.global_variables_initializer()  

X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]]) # t = 0 
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]]) # t = 1 

with tf.Session() as sess: 
	init.run() 
	Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})
	print(Y0_val, Y1_val)  

#to see that all of these functions mean and carry out look at page 387. 

#Packing sequences: 

tf.reset_default_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5 

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2])) 
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons) 
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype = tf.float32) 
outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])

X_batch = np.array([
	[[0,1,2], [9,8,7]],	#Instance 1 
	[[3,4,5], [0,0,0]], #Instance 2
	[[6,7,8], [6,5,4]],	#Instance 3
	[[9,0,1], [3,2,1]],	#Instance 4
])

init = tf.global_variables_initializer() 

with tf.Session() as sess: 
	init.run() 
	outputs_val = outputs.eval(feed_dict={X:X_batch})
	print(outputs_val)  

#And we get a single output tensor for all time steps and 
#all neurons. However this approach still builds a graph containing one cell
#per time step. If there were 50 time steps, the graph would look pretty ugly.

##Dynamic unrolling through time: 
#The dynamic_runn() function uses a while loop operation to run over the cell 
#the appropriate number of times. Conveniently, it also accepts a single tensor 
#for all inputs at every time step and it outputs a single tensor for all 
#outputs at every time step; there is no need to stack, unstack, or transpose. 

tf.reset_default_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5  

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons) 
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32) 

init = tf.global_variables_initializer() 

X_batch = np.array([
	[[0,1,2], [9,8,7]],	#Instance 1 
	[[3,4,5], [0,0,0]], #Instance 2
	[[6,7,8], [6,5,4]],	#Instance 3
	[[9,0,1], [3,2,1]],	#Instance 4
])

with tf.Session() as sess: 
	init.run() 
	outputs_val = outputs.eval(feed_dict={X: X_batch}) 
	print(outputs_val) 

##Handling Variable Length input sequences: 
#What if the input sequences have variable lengths (like sentences)?
#In this case you should set the sequence_length arguement when calling the dynamic_rnn() 
#function it must be 1d tensor indicating the length of the input sequence 
#for each instance.

tf.reset_default_graph() 

n_step = 2
n_inputs = 3
n_neurons = 5 

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) 

seq_length = tf.placeholder(tf.int32, [None]) #Interesting this holds all of the 
#sequence length values from a the predictor variables. 
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32, 
									sequence_length=seq_length) 

init = tf.global_variables_initializer() 

X_batch = np.array([
	[[0,1,2], [9,8,7]],	#Instance 1 
	[[3,4,5], [0,0,0]], #Instance 2
	[[6,7,8], [6,5,4]],	#Instance 3
	[[9,0,1], [3,2,1]],	#Instance 4
])
seq_length_batch = np.array([2,1,2,2]) #Now I understand the second value is 
#1 because within instance 2 the contents in one of the strings is only 
#conposed of zeros. 

with tf.Session() as sess: 
	init.run() 
	outputs_val, states_val = sess.run(
		[outputs, states], feed_dict={X: X_batch, seq_length:seq_length_batch})
		#Remember that since seq_length is a placeholder both the seq_length and 
		#the X variables need to be entered into the computation graph. 
	print(outputs_val)
	print(states_val) 

##Handling Variable length output sequences: 
#What if the output sequences have variable lengths as well? In this case the most 
#common solution is to define a special output called an end of sequence token (EOS token) 
#Any output past the EOS should be ignored.

##Training RNNs:
#To train an RNN, the trick is to unroll it through time (like we just did) and then 
#simply use regular backpropagation. this stragey is called backpropagation through 
#time. 

##Training a sequence Classifier: 
#the author will use the mnist dataset again with an architecture of 
#150 recurrent neurons, plus a fully connected layer containing 10 neurons 
#(one per class) connected to the output of the last time step, followed by a 
#softmax layer. 

tf.reset_default_graph() 

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split  

n_steps = 28 
n_inputs = 28 
n_neurons = 150 
n_outputs = 10 

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, 
															random_state = 42, shuffle = True)
X_train = X_train.reshape((-1, n_steps, n_inputs))  
X_test = X_test.reshape((-1, n_steps, n_inputs)) 
y_train = y_train 
y_test = y_test
print(X_train.shape) 

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape = [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.int32, shape = [None]) 

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons) 
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32) 

logits = tf.layers.dense(states, n_outputs)   
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits) 

loss = tf.reduce_mean(xentropy) 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(loss) 
correct = tf.nn.in_top_k(logits, y, 1) 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer() 

n_epochs = 5 
batch_size = 150
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))
print(n_batches)  

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
			sess.run(training_op, feed_dict = {X: X_batch, y:y_batch}) 
		acc_train = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict = {X: X_test, y:y_test}) 
		print(epoch, "Train accuracy:", acc_train, "Test_accuracy:", acc_test)
#Sweet!!! At least this code snippet works. Interestingly this method starts 
#with a very low training accuracy rate and the training rate seems to be diverging 
#from the global minimum. Will need to look into what the bug might be. 

#there has to be a bug some where in the code. the accuracy ratings are way off from 
#what the author obtained. 

#Two possible problem locations the tf.contrib.rnn.BasicRNNCell() might be 
#broken or the input data obtained from sklearn.datasets doesn't work with 
#this application. 

##Training to perdict time series: 
#In this section we will train an RNN to predict the next value in a generated 
#time series. Each training instance is a randomly selected sequence of 20 
#consecutive values from the time series, and the target sequence is the same as the input 
#sequence, except it is shifted by one time step into the future.

#time series dataset: 
t_min, t_max = 0, 30 
resolution = 0.1 

def time_series(t):
	return t * np.sin(t) / 3 + 2 * np.sin(t*5) 

def next_batch(batch_size, n_steps):
	t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution) 
	Ts = t0 + np.arange(0., n_steps + 1) * resolution 
	ys = time_series(Ts) 
	return ys[:, : - 1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1) 

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution)) 

n_steps = 20 
t_instance = np.linspace(start = 12.2, stop = 12.2 + resolution * (n_steps + 1), num = n_steps + 1, endpoint = True)  
print(t_instance)  
#First let's create a RNN. It will contain 100 recurrent neurons and we will 
#unroll it over 20 time steps since each training instance will be 20 inputs 
#long. Each input will contain only one feature. The targets are also sequences 
#of 20 inputs, each containing a single value. 

tf.reset_default_graph() 

n_steps = 20 
n_inputs = 1 
n_neurons = 100 
n_outputs = 1 

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs]) 
cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu) 
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32) 

#At each time step we now have an output vector of size 100. But we actually want is 
#a single output value at each time step. The simplest solution is to wrap the 
#cell in an OutputProjectionWrapper. 

#Time series illustration using the wrapper:

tf.reset_default_graph() 

n_steps = 20 
n_inputs = 1 
n_neurons = 100 
n_outputs = 1 

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs]) 

cell = tf.contrib.rnn.OutputProjectionWrapper( 
	tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu), 
	output_size = n_outputs) 
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32) 

#creating a cost function and an optimization value:
learning_rate = 0.001 

loss = tf.reduce_mean(tf.square(outputs - y)) 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer() 

saver = tf.train.Saver() 

n_iterations = 1500 
batch_size = 50 

with tf.Session() as sess: 
	init.run() 
	for iteration in range(n_iterations): 
		X_batch, y_batch = next_batch(batch_size, n_steps) 
		sess.run(training_op, feed_dict={X: X_batch, y:y_batch}) 
		if iteration % 100 == 0:
			mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
			print(iteration, "\tMSE:", mse) 
	saver.save(sess, "./my_time_series_model") 

with tf.Session() as sess:
	saver.restore(sess, "./my_time_series_model") 

	X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
	y_pred = sess.run(outputs, feed_dict = {X:X_new}) 

print(y_pred)

#graphic representation: 
import matplotlib.pyplot as plt 

plt.title("Testing the model", fontsize = 14) 
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize = 10, label = "instance") 
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize = 10, label = "target") 
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize = 10, label = "prediction") 
plt.legend(loc="upper left") 
plt.xlabel("time") 
plt.show()

##Creative RNN: 
#Watching the algorithm create something new with the time series 
#model that we just trained it with. 

with tf.Session() as sess: 
	saver.restore(sess, "./my_time_series_model") 

	sequence = [0.] * n_steps# seed sequence it start the algorithm 
	for iteration in range(300):
		X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1) 
		y_pred = sess.run(outputs, feed_dict = {X:X_batch}) 
		sequence.append(y_pred[0,-1, 0])

plt.figure(figsize=(8,4))
plt.plot(np.arange(len(sequence)), sequence, "b-") 
plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3) 
plt.xlabel("time") 
plt.ylabel("value") 
plt.show()     

##Deep RNNs: 
#to implement a deep RNN in tensorflow, you can create several cells and stack 
#them into a multiRNNCell. 

#The states variable is a tuple containing one tensor per layer, 
#each representing the final state of that layer's cell (with shape 
#[batch_size, n_neurons]). 
tf.reset_default_graph() 

n_steps = 28 
n_inputs = 28 
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, shape = [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.int32, shape = [None]) 

n_neurons = 100 
n_layers = 3 

layers = [tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, 
	activation = tf.nn.relu)
	for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers) 
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32) 

states_concat = tf.concat(axis = 1, values = states) 
logits = tf.layers.dense(states_concat, n_outputs) 

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits) 
loss = tf.reduce_mean(xentropy) 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(loss) 
correct = tf.nn.in_top_k(logits, y, 1) 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer() 

n_epochs = 10 
batch_size = 150
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))
print(n_batches)  

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
			sess.run(training_op, feed_dict = {X: X_batch, y:y_batch}) 
		acc_train = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict = {X: X_test, y:y_test}) 
		print(epoch, "Train accuracy:", acc_train, "Test_accuracy:", acc_test)
#This model is way better than the first one, but the only problem is that the training 
#time is a little long. Will need to think about dimensional reduction methods 
#to increase the training time. The algorithm seems to be converging to the global
#minimum. 

#The final accuracy rate for the model was a 97 percent test accuracy rate.  

##Applying Dropout: 
#If you build a very deep RNN, it may end up overfitting the training set. To 
#prevent this, a common technique is to use the drop out method before or after the 
#RNN layers. 

#As a means to use a drop out layer inbetween RNN layers you can 
#use the wrapper DropoutWrapper(). 

tf.reset_default_graph() 

n_inputs = 1 
n_neurons = 100 
n_layers = 3
n_steps = 20 
n_outputs = 1 

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs]) 
keep_prob = tf.placeholder_with_default(1.0, shape = ())
cells = [tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
		for layer in range(n_layers)] 

cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
				for cell in cells] 
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop) 
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32) 
learning_rate = 0.01 
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs) 
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs]) 

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)   
training_op = optimizer.minimize(loss) 

init = tf.global_variables_initializer() 
saver = tf.train.Saver() 

n_iterations = 1500 
batch_size = 50 
train_keep_prob = 0.5 

with tf.Session() as sess: 
	init.run() 
	for iteration in range(n_iterations): 
		X_batch, y_batch = next_batch(batch_size, n_steps) 
		_, mse = sess.run([training_op, loss], feed_dict =
						{X: X_batch, y: y_batch, keep_prob: train_keep_prob})

		if iteration % 100 == 0:
			print(iteration, "Training MSE:", mse) 

	saver.save(sess, "./my_dropout_time_series_model") 

with tf.Session() as sess: 
	saver.restore(sess, "./my_dropout_time_series_model") 

	X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
	y_pred = sess.run(outputs, feed_dict = {X: X_new}) 

plt.title("Testing the model", fontsize = 14) 
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize = 10, label = "instance") 
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize = 10, label = "target") 
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize = 10, label = "prediction") 
plt.legend(loc="upper left") 
plt.xlabel("time") 
plt.show()

##The difficulty of training over many time steps:
#To train an RNN on long sequences you will need to run it over many time 
#steps making the unrolled RNN a very deep network. Just like any deep neural 
#network it may suffer from the vanishing/exploding gradient problem. Solutions 
#to these problems include (of course) gradient clipping, nonsaturated activation functions,
#dropout method, and batch normalization. 

#The simplest and most common solution to this problem is to unroll 
#the RNN only over a limited number of time steps during training (truncated 
#backpropagation through time). the problem, of this solution is that 
#the model will not be able to learn long-term patterns. 

#Besides the long training time, a second problem faced by long-running RNNs is the 
#fact that the memory of the first inputs gradually fades away. Meaning that the first 
#inputs will be virtually deleted from the network during the running 
#of an extremely long recurrent neural network. 

