###Hands on machine learning with Scikit learning and tensorflow
###chapter 13 exercises:

##1.) The advantages of Convolutional neural networks over the more 
#traditional deep neural network architectures are:

#First for complex image recognition tasks, CNNs are able to detect complex patterns 
#within large image files with more precision and less features per 
#hidden layer. 

#Second the way convolutional layers are set up (in the realm that each 
#neuron doesn't need to be attached to a neuron in the succeeding layer) 
#allows for faster training speed for image recognition tasks and in 
#addition the discovery of horizontal and vertical convolutional layers 
#allows for greater image classification precision and scalability. 
#Better scalability because when utilizing the classical deep neural 
#network architectures for complex image classification problems the 
#trained classifier can only detect target object classes if: one, they 
#look exactly the same as the training examples and two, the target targets 
#are located in exactly the same place as the training examples. Hence 
#deep neural networks can only be used on simple image classification 
#tasks (like the mnist dataset). 

##Author's answer: 
#these are the main advantages of a CNN over a fully connected DNN for image 
#classification: 

#Because consecutive layers are only partially connected and because it heavily 
#reuses its weights, a CNN has many fewer parameters than fully connected
#DNN, which makes it much faster to train, reduces the risk of overfitting,
#and requires much less training data. 

#When a CNN has learned a kernel that can detect a particular feature, it 
#can detect that feature anywhere on the image. In contrast, when a DNN learns a feature in 
#one location, it can detect it only in that particular location. Hence this 
#leads to better scalability for the CNN method.

#Finally a DNN has no prior knowledge of how pixels are organized; it does 
#not know that nearby pixels are close. A CNN's architecture embeds this prior knowledge 
#Lower layers typically identify features in small areas of the images, while 
#higher layers combine the lower-level features into larger features. this 
#works well with most natural images, giving CNNs a decisive head start compared to 
#DNNs.

##2.) author's solution: Since the question is asking me to compute computational 
#expense with a convolutional neural network architecture of three total 
#convolution layers. 

#Let's compute how many parameters the CNN has. Since its first convolutional layer has 
#3 by 3 kernels, and the input has three channels, then each feature map
#has 3 *3 * 3 weights, plus a bias term. There's 28 parameters per feature map.
#Since this first convolutional layer has 100 feature maps, it has a total of 
#2,800 parameters. The second convolutional layer has 3 * 3 kernels, and its input is the 
#set of 100 feature maps of the previous layer, so each feature map has 3 * 3 * 100 
#weights, plus a bias term. Since it has 200 feature maps, this layer has 901 * 200 
#parameters. The third convolutional layer has 3 * 3 kernels, and it's input is the set of 200 
#features maps of the previous layers, so each feature map has 3 * 3 * 200 weights plus bias terms.
#Since it has 400 feature maps, this layer has a total of 1801 * 400 
#parameters. All in all, the CNN has 903,400 total parameters. 

##3.) The things to do if one runs out of gpu memory during training a 
#convolutional neural network: 

#One, the most apparent courses of action is to use the technique descussed 
#in chapter 12, which are connecting multiple gpu nodes onto your personal 
#device. You can use a multi device setup, as again descussed in chapter 12 but 
#the computational delay apparent within such architectures will only increase 
#training time. Hence using multiple gpus is by far the most straight forward 
#solution. 

#Two, A practicioner can train the model through specific epoch windows and 
#save the weights and bias terms within specified checkpoints within the tensorflow 
#session call. This won't really fix the problem of slow convergence and the fear 
#of running out of ram during training, but through knowing the limitations of one's 
#device before hand you can create an effective schedule that can lead to 
#a fully trained model over the course of a couple hours or in some cases days.

#Three, since the CNN setup claims so much memory during training, I believe that 
#cutting one's training set into smaller mini batches might help with memory limitations

#Four, use 16-bit floats in place of 32-bit floats.

#Five, remove one or more layers. 

##4.) One will add a pooling layer in place of another convolutional 
#layer if the practioner wants to shrink the feature map to a more 
#manageable shape (or rather size). In other words, convolutional layers 
#increase the feature map while pooling layers aggregate the input feature 
#map through either summing or averaging the values. 

#Author's answer: A max pooling layer has no parameters at all, whereas a 
#convolutional layer has quite a few.

##5.) A local response normalization layer makes the neurons that most strongly 
#activate inhibit neurons at the same location but in neighboring feature maps 
#which encourages different feature maps to specialize and pushes them apart,
#forcing them to explore a wider range of features. It is typically used in 
#the lower layers to have a larger pool of low-level features that the upper 
#layers can build upon.

##6.) the inovations that were established by the AlexNet model compared to 
#the original convolutional neural network architecture (LeNet-5) are 
#the former used stacked convolutional layers in place of the traditional 
#pooling layer after each convolutional layer configuration. In addition,
#for the final three layers, the team used the ReLU activation function with 
#the drop out method (in place of batch normalization). 

#In addition, the AlexNet architecture uses the local response normalization method 
#after the first convolutional layer. This form of normalization makes the neurons 
#that most strongly activate inhibit neurons at the same location but in 
#neighboring feature maps. this encourages different feature maps to specialize 
#pushing them apart and forcing them to explore a wider range of features, ultimately 
#improving generalization.

#As for the main innovations for the GoogLeNet architecture, the teams 
#discovery of inception modules greatly improved parameter scaling with 
#10 times fewer features than the AlexNet architecture. These 1 by 1 
#inception layers can be considered as bottleneck layers thus helping with 
#mitigating overfitting with a side affect of creating more powerful 
#classification layers that can achieve a greater rate of accuracy.

#As for the innovations of the ResNet architecture, inplace of finding 
#the optimum values of a mathematical hypothesis (as defined by Andrew Ng as 
#h_theta(x)) the research team created an algorithm that uses skip connections 
#to force the network to model f(x) = h(x) - x. The end result is called 
#residual learning. 

##7.) 
import numpy as np 
from sklearn.datasets import load_sample_images 
import tensorflow as tf 
import matplotlib.pyplot as plt 

channels = 3
filters = np.zeros(shape = (7,7, channels, 2), dtype = np.float32) 
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1 

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)  
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train = y_train.astype(np.int32) 
y_test = y_test.astype(np.int32) 

height = 28 
width = 28 
channels = 1 
n_inputs = height * width 

conv1_fmaps = 32 
conv1_ksize = 3
conv1_stride = 1 
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps 

n_fcl = 64
n_outputs = 10  

with tf.name_scope("input"):
	X = tf.placeholder(tf.float32, shape = [None, n_inputs], name = "X") 
	X_reshaped = tf.reshape(X, shape = [-1, height, width, channels]) 
	y = tf.placeholder(tf.int32, shape = [None], name = "y") 

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size = conv1_ksize,
						strides=conv1_stride, padding=conv1_pad,
						activation=tf.nn.relu, name = "conv1") 
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
						strides=conv2_stride, padding=conv2_pad, 
						activation=tf.nn.relu, name="conv2") 

with tf.name_scope("pool3"):
	pool3 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID") 
	pool3_flat = tf.reshape(pool3, shape = [-1, pool3_fmaps * 7 * 7]) 

with tf.name_scope("fcl"):
	fcl = tf.layers.dense(pool3_flat, n_fcl, activation = tf.nn.relu, name = "fcl") 

with tf.name_scope("output"):
	logits = tf.layers.dense(fcl, n_outputs, name = "output") 
	Y_proba = tf.nn.softmax(logits=logits, name= "Y_proba") 

with tf.name_scope("train"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)  
	loss = tf.reduce_mean(xentropy) 
	optimizer = tf.train.AdamOptimizer() 
	training_op = optimizer.minimize(loss) 

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 

with tf.name_scope("init_and_save"):
	init = tf.global_variables_initializer() 
	saver = tf.train.Saver() 

n_epoches = 1
batch_size = 100 
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epoches):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) 
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) 
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test}) 
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test) 

		save_path = saver.save(sess, "./my_mnist_model")
#this convolutional network implementation was created by the author. Just like 
#what the author said about convolutional neural networks, training is remarkably 
#slow. will need to see if overfitting the training set is truely a problem 
#with this method.

tf.reset_default_graph() 

#LeNet-5 architecture implementation using the mnist dataset:
height = 28 
width = 28 
channels = 1 
n_inputs = height * width 


with tf.name_scope("input"):
	X = tf.placeholder(tf.float32, shape = [None, n_inputs], name = "X") 
	X_reshaped = tf.reshape(X, shape = [-1, height, width, channels]) 
	y = tf.placeholder(tf.int32, shape = [None], name = "y") 

with tf.name_scope("CNN"):
	conv1 = tf.layers.conv2d(X_reshaped, filters=6, kernel_size = [5,5], 
							strides = 1, activation = tf.nn.tanh, padding = "VALID",
							name = "conv1") 
	pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "VALID")
	pool1_flat = tf.reshape(pool1, shape = [-1, 6 * 7 * 7])      
	fcl1 = tf.layers.dense(pool1_flat, 86, activation=tf.nn.tanh, name = "fcl1")
	logits = tf.layers.dense(fcl1, 10, name = "output") 
	#It seems that I can't stack pooling layers on top of one another. most likely 
	#I'm getting the tensor scaling wrong. Will need to come back to this problem later.

with tf.name_scope("train"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)  
	loss = tf.reduce_mean(xentropy) 
	optimizer = tf.train.AdamOptimizer() 
	training_op = optimizer.minimize(loss) 

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


with tf.name_scope("init_and_save"):
	init = tf.global_variables_initializer() 
	saver = tf.train.Saver() 

n_epoches = 5
batch_size = 100 
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epoches):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) 
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) 
		acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test}) 
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test) 

		save_path = saver.save(sess, "./my_mnist_model")




