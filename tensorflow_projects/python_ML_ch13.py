###Hands on machine learning with Scikit learning and tensorflow
### Chapter 13 Convolutional neural networks:
##convolutional layers: the most important building block of a CNN is the 
#convolutional layer. neurons in the first convolutional layer is not connected to every single 
#pixels in the image (much like the neural network architecture used 
#for the mnist dataset), but only to pixels in their receptive fields.

#Interesting addition, in place of transforming the input data into a 1 dimensional 
#vector convolutional neural networks only accept 2 dimensional data. 

#A neuron located in row i, column j of a given layer connected to the output 
#of the neurons in the previous layer located in rows i to i + f_h - 1, columns j 
#to j + f_w - 1, where f_h and f_w is are the height and width of the receptive field.
#In order for a layer to have the same height and width as the previous layer, it is 
#common to add zeros around the inputs (called zero padding).

#It is also passible to connect a large input layer to a much smaller layer by 
#spacing out the receptive fields. the distance between two consecutive receptive fields 
#is called the stride. A neuron located in row i, column j in the upper layer is connected 
#to the outputs of the neurons in the previous layer located in rows i * s_h to i * s_h + f_h -1,
#columns j * s_w + f_w - 1, where s_h and s_w are the vertical and horizontal strides. 

##Filters: 
#A neuron's weight can be represented as a small image the size of the receptive 
#field. The first one is a representated as a black square with a veritcal white line
#in the middle (it is a 7 by 7 matrix full of 0s except for the central column, which is full of 1s);
#neurons using these weights will ignore everything in their receptive field except for the central vertical 
#line. the second filter is a black square with a horizontal white line in the middle.
#Once again, neurons using these weights will ignore everything in their receptive field 
#except for the central horizontal line. 

##Stacking multiple feature maps:
#Up to now, for simplicity, we have represented each convolutional layer as a thin 2D
#layer, but in reality it is composed of several feature maps of equal sizes, so it is more 
#accurately represented in 3D. Within one feature map, all neurons share the same parameters 
#(weights and bias term), but different feature maps may have different 
#parameters. A neuron's receptive field is the same as described earlier,
#but it extends across all the previous layers' feature maps. 

#Moreover, input images are also composed of multiple sublayers: one per color channel.
#Grey scale images only have one color channel. 

#Specifically, a neuron located in row i, column j of the feature map k in a given 
#convolutional layer l is connected to the outputs of the neuron in the previous layer
#l - 1, located in rows i * s_w to i * s_w f_w -1 and columns j * s_h to 
#j * s_h + f_h - 1, across all feature maps (in layer l-1 ) 

##Tensorflow implementation: 
#In tensorflow, each input image is typically represented as a 3D tensor of shape
#[height, width, channels]. A mini batch is represented as a 4D tensor of shape 
#[mini-batch size, height, width, channels]. The weights of a convolutional 
#layer are represented as a 4D tensor of shape [f_h, f_w, f_n, f_n'].
#the bias terms of a convolutional layer are simply represented as a 1D tensor of shape 
#[f_n].

#Through the tensorflow api you can create convolutional layers through the command 
#tf.nn.conv2d() in place of tf.layer.dense() will need to look into how 
#to create a convolutional layer from scratch. 

import numpy as np 
from sklearn.datasets import load_sample_images 
import tensorflow as tf 
import matplotlib.pyplot as plt 

#load sample images:
dataset = load_sample_images()
china = dataset.images[0] 
flower = dataset.images[1] 
dataset = np.array([china, flower], dtype = np.float32) 
batch_size, height, width, channels = dataset.shape 

#create 2 filters:
filters = np.zeros(shape = (7,7, channels, 2), dtype = np.float32) 
filters[:, 3, :, 0] = 1#vertical line 
filters[3, :, :, 1] = 1 #horizontal line 

#Create a graph with input X plus a convolutional layer applying the 2 filters 
X = tf.placeholder(tf.float32, shape = (None, height, width, channels)) 
convolution = tf.nn.conv2d(X, filters, strides = [1, 2, 2, 1], padding = "SAME") 

with tf.Session() as sess:
	output = sess.run(convolution, feed_dict = {X: dataset}) 

plt.imshow(output[0, :, :, 1], cmap = "gray") #plot 1st image's 2nd feature map.
plt.show()#No way this actually worked. Let's see what it looks like with only one 
#convolutional filter running. 
tf.reset_default_graph() 

#vertical: 
vertical = np.zeros(shape = (7,7, channels, 2), dtype = np.float32)
vertical[:, 3, :, 0] = 1 
vertical[3, :, :, 1] = 1 
X = tf.placeholder(tf.float32, shape = (None, height, width, channels)) 
convolution = tf.nn.conv2d(X, vertical, strides = [1, 2, 2, 1], padding = "SAME")
with tf.Session() as sess:
	output = sess.run(convolution, feed_dict = {X: dataset}) 

def plot_image(image):
	plt.imshow(image, cmap = "gray", interpolation = "nearest") 
	plt.axis("off")

def plot_color_image(image):
	plt.imshow(image.astype(np.uint8), interpolation = "nearest") 
	plt.axis("off") 

for image_index in (0, 1):
	for feature_map_index in (0, 1):
		plot_image(output[image_index, :, :, feature_map_index]) 
		plt.show()#Can't seem to get the image to show up. Will need to look into 
#what I'm doing wrong with the convolutional layer commands. 
tf.reset_default_graph() 

#arguments within the tf.nn.conv2d() command:

#X is the input mini-batch

#filters is the set of filters to apply

#strides is a four element 1D array, where the two central elements are the vertical 
#and horizontal strides. the first and last elements must currently be equal to 1.

#padding must be either valid or same:
#valid, the convolutional layer does not use zero padding, and may 
#ignore some rows and columns at the bottom and right of the input image,
#depending on the stride.
#Same, the convolutional layer uses zero padding if necessary.

#In a real CNN you would let the training algorithm discover the best filters 
#to automatically discover the best filters. Tensorflow has a 
#tf.layers.conv2d() function which creates the filters variable for you 
#(called the kernel) and initializes it for you. 

X = tf.placeholder(shape = (None, height, width, channels), dtype = tf.float32) 
conv = tf.layers.conv2d(X, filters = 2, kernel_size = 7, strides = [2,2], 
	padding = "SAME")

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run() 
	output = sess.run(conv, feed_dict = {X: dataset}) 

plot_color_image(output[0, :, :, 1]) 
plt.show()
tf.reset_default_graph() 

#Unfortunately, convolutional layers have quite a few hyperparameters: you must choose
#the number of filters, their height and width, the strides, and the padding, type. 

#Valid vs Same padding:
filter_primes = np.array([2., 3., 5., 7., 11., 13.], dtype = np.float32) 
x = tf.constant(np.arange(1, 13 + 1, dtype=np.float32).reshape([1, 1, 13, 1]))
filters = tf.constant(filter_primes.reshape(1, 6, 1, 1))

valid_conv = tf.nn.conv2d(x, filters, strides = [1,1,5,1], padding = "VALID") 
same_conv = tf.nn.conv2d(x, filters, strides = [1,1,5,1], padding = "SAME") 

with tf.Session() as sess:
	print("VALID:\n", valid_conv.eval()) 
	print("Same:\n", same_conv.eval())

tf.reset_default_graph() 

##Pooling layer: 
#Their goal is to subsample the input image in order to reduce the computational
#the memory usage, and the number of parameters. Reducing the input image size
#also makes the neural network tolerate a little bit of image shift 

#Just like in convolutional layers, each neuron in a pooling layer is connected to the 
#outputs of a limited number of neurons in the previous layer, located within a small 
#rectangular receptive field. You must define the size, the stride, and the padding 
#type. However, a pooling neuron has no weights all it does is aggregate the 
#inputs using an aggregate function such as the max and mean. 

#tensorflow implementation: 
filters = np.zeros(shape=(7,7, channels, 2), dtype = np.float32) 
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1 

X = tf.placeholder(tf.float32, shape = (None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID") 

with tf.Session() as sess:
	output = sess.run(max_pool, feed_dict = {X: dataset}) 

plt.imshow(output[0].astype(np.uint8))
plt.show()

tf.reset_default_graph() 
#I still can't seem to get the same vibrant colors as the author 
#Still though this image does seem better than the other method (or rather 
#the convolution layer implementation). 
#Now the colors seem to be vibrant I believe that using the command output[0, :, :, 1] 
#distorts the color accuracy of colored images. Will need to look into this.

#The ksize argument contains the kernel shape along all four dimensions of the input 
#tensor: [batch size, height, width, channels]. Tensorflow currently does not support 
#pooling over multiple instances.

#To create an average pooling layer, just use the avg_pool() function instead of 
#max_pool().

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

n_epoches = 1000
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




















