###Pro deep learning chapter 5 part 2:

##A Restricted Boltzmann Implementation in Tensorflow: 
#Implemenation of restricted boltzmann machines using the mnist dataset. 
#I believe that the end result can be likened to that of using auto encoders 
#or even PCA machine learning methods. 

import tensorflow as tf 
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

def read_infile():
	n_classes = 10
	mnist = fetch_mldata("MNIST original") 
	X_mnist, y_mnist = mnist["data"], mnist["target"]
	y_mnist = np.eye(n_classes)[y_mnist.astype(np.int32)]  
	X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)    
	return X_train, y_train, X_test, y_test 

X_train, y_train, X_test, y_test = read_infile()

n_visible = 784 
n_hidden = 500 
display_step = 1 
num_epochs = 15 
batch_size = 256 
lr = tf.constant(0.001, tf.float32) 

#Creating the computational graph for the tensorflow execution phase:
x = tf.placeholder(tf.float32, [None, n_visible], name="x") 
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name = "W") 
b_h = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name = "b_h"))#The bias vector 
#for the hidden layer 
b_v = tf.Variable(tf.zeros([1, n_visible], tf.float32, name = "b_v"))#the bias vector for 
#the visible layer. 

#Converts the probability into discrete binary states, 0 and 1:
def sample(probs):
	return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)) 

#Gibbs sampling step: 
def gibbs_step(x_k):
	h_k = sample(tf.sigmoid(tf.matmul(x_k, W) + b_h)) 
	x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_v)) 
	return x_k 

#Run multiple gibbs sampling steps starting from an initial point:
def gibbs_sample(k, x_k):
	for i in range(k):
		x_out = gibbs_step(x_k) 
	#returns the gibbs sample after k iterations:
	return x_out 

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

#Constrastive divergence algorithm: 
x_s = gibbs_sample(2,x) 
h_s = sample(tf.sigmoid(tf.matmul(x_s, W) + b_h)) 
#sample hidden states given visible states
h = sample(tf.sigmoid(tf.matmul(x, W) + b_h))
#sample visible states based on hidden states 
x_ = sample(tf.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v))

#Weight updated based on gradient descent:
size_batch = tf.cast(tf.shape(x)[0], tf.float32) 
W_add = tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(x), h), 
	tf.matmul(tf.transpose(x_s), h_s)))
bv_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(x, x_s), 0, True)) 
bh_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(h, h_s), 0, True)) 
updt = [W.assign_add(W_add), b_v.assign_add(bv_add), b_h.assign_add(bh_add)]

##Tensorflow graph execution phase:
with tf.Session() as sess:
	#Initialize the variables of the model: 
	init = tf.global_variables_initializer() 
	sess.run(init) 

	total_batch = int(X_train.shape[0]/batch_size) 
	#start the training 
	for epoch in range(num_epochs):
		#loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = fetch_batch(epoch, i, batch_size) 
			#Run the weight update
			batch_xs = (batch_xs > 0)*1
			_ = sess.run([updt], feed_dict={x:batch_xs}) 
		#Display the running step 
		if epoch % display_step == 0:
			print("Epoch:", "%04d" % (epoch + 1))

	print("RBM training Completed !") 

	##Generate hidden structure for 1st 20 images in test mnist

	out = sess.run(h, feed_dict={x:(X_test[:20]>0)*1})
	label = y_test[:20]

	#Take the hidden representation of any of the test images 
	#The output level of the 3rd record should match the image generated
	plt.figure(1) 
	for k in range(20):
		plt.subplot(4, 5, k+1)
		image = (X_test[k]>0)*1
		image = np.reshape(image, (28,28))
		plt.imshow(image, cmap="gray") 
		plt.show()

	plt.figure(2) 
	for k in range(20):
		plt.subplot(4, 5, k+1) 
		image = sess.run(x_, feed_dict={h:np.reshape(out[k],(-1, n_hidden))})
		image = np.reshape(image, (28,28))
		plt.imshow(image, cmap='gray') 
		print(np.argmax(label[k]))
		plt.show()

##Deep belief networks (DBNs) implemenation using tensorflow and 
#again the mnist dataset.

tf.reset_default_graph()  
n_visible = 784 
n_hidden = 500 
display_step = 200 
num_epochs = 200 
batch_size = 256
num_batches = int(np.ceil(int(X_train.shape[0]) / batch_size)) 
lr = tf.constant(0.001, tf.float32) 
learning_rate_train = tf.constant(0.01, tf.float32) 
n_classes = 10 
training_iters = 200 

#Defining the computation graph:
x = tf.placeholder(tf.float32, [None, n_visible], name="x") 
y = tf.placeholder(tf.float32, [None, 10], name='y') 

W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") 
b_h = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name = "b_h"))
b_v = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="b_v"))
W_f = tf.Variable(tf.random_normal([n_hidden, n_classes], 0.01), name="W_f") 
b_f = tf.Variable(tf.zeros([1, n_classes], tf.float32, name='b_f'))

#Converts the probability into discrete binary states, 0 and 1:
def sample(probs):
	return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)) 

#Gibbs sampling step: 
def gibbs_step(x_k):
	h_k = sample(tf.sigmoid(tf.matmul(x_k, W) + b_h)) 
	x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_v)) 
	return x_k 

#Run multiple gibbs sampling steps starting from an initial point:
def gibbs_sample(k, x_k):
	for i in range(k):
		x_out = gibbs_step(x_k) 
	#returns the gibbs sample after k iterations:
	return x_out 

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

#Constrastive divergence algorithm: 
x_s = gibbs_sample(2,x) 
h_s = sample(tf.sigmoid(tf.matmul(x_s, W) + b_h)) 
#sample hidden states given visible states
h = sample(tf.sigmoid(tf.matmul(x, W) + b_h))
#sample visible states based on hidden states 
x_ = sample(tf.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v))

#Weight updated based on gradient descent:
size_batch = tf.cast(tf.shape(x)[0], tf.float32) 
W_add = tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(x), h), 
	tf.matmul(tf.transpose(x_s), h_s)))
bv_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(x, x_s), 0, True)) 
bh_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(h, h_s), 0, True)) 
updt = [W.assign_add(W_add), b_v.assign_add(bv_add), b_h.assign_add(bh_add)]

#OPS for the classification network:
h_out = tf.sigmoid(tf.matmul(x, W) + b_h) 
logits = tf.matmul(h_out, W_f) + b_f
prob = tf.nn.softmax(logits) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_train).minimize(cost) 
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

#Tensorflow graph execution:
with tf.Session() as sess: 
	init = tf.global_variables_initializer() 
	sess.run(init) 

	total_batch = int(X_train.shape[0]/batch_size) 

	#start training 
	for epoch in range(num_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = fetch_batch(epoch, i, batch_size)
			#run the weight update
			batch_xs = (batch_xs > 0)*1 
			_ = sess.run([updt], feed_dict={x:batch_xs})

		#Display the running step 
		if epoch % display_step == 0:
			print("Epoch", "%04d" % (epoch+1))

	print("RBM training completed!") 

	out = sess.run(h, feed_dict={x:(X_test[:20]>0)*1})
	label = y_test[:20]

	plt.figure(1) 
	for k in range(20):
		plt.subplot(4, 5, k+1)
		image = (X_test[k]>0)*1
		image = np.reshape(image, (28,28))
		plt.imshow(image, cmap="gray") 
	plt.show()

	plt.figure(2) 
	for k in range(20):
		plt.subplot(4, 5, k+1) 
		image = sess.run(x_, feed_dict={h:np.reshape(out[k],(-1, n_hidden))})
		image = np.reshape(image, (28,28))
		plt.imshow(image, cmap='gray') 
		print(np.argmax(label[k]))
	plt.show()

	#Invoke the classification network training now:
	for batch_index in range(training_iters):
		batch_x, batch_y = fetch_batch(1, batch_index, batch_size)
		sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
		if i % 10 == 0:
			#Calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y}) 
			print("Iter" + str(i) + ", Minibatch Loss=" + "\n{:.6f}".format(loss) +
					", Training Accuracy= " + "\n {:.5f}".format(acc)) 
	print("Optimization finished!") 

	#Calculate accuracy for 256 mnist test images
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:X_test[:256], y:y_test[:256]}))



