###Pro deep learning chapter 5 part 3:

##Deep belief networks (DBNs) implemenation using tensorflow and 
#again the mnist dataset.

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
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

	#Invoke the classification network training now:
	for epoch in range(num_epochs):

		for batch_index in range(total_batch):
			batch_x, batch_y = fetch_batch(epoch, batch_index, batch_size)
			sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
			if batch_index % 10 == 0:
				#Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y}) 
				print("Iter" + str(batch_index) + ", Minibatch Loss=" + "\n{:.6f}".format(loss) +
					", Training Accuracy= " + "\n {:.5f}".format(acc)) 
	print("Optimization finished!") 

	#Calculate accuracy for 256 mnist test images
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:X_test[:256], y:y_test[:256]}))


##Sparse Auto-encoder implemenation in Tensorflow:
#Again this representation uses the mnist dataset to show implemenation of 
#a sparse autoencoder that uses the KL diverence as the regularization term within 
#the cost function. 

#Parameters for training the network 
tf.reset_default_graph() 

learning_rate = 0.001 
training_epochs = 200 
batch_size = 126
display_step = 1 
examples_to_show = 10

#computation map construction phase:
#remember that the hidden layer dimensions need to be larger than the dimensions of the 
#input layer. Thus meaning in this case the hidden layer will have 32*32 units 
#and the input layer will have only 28*28 units.
n_hidden_1 = 32*32 
n_input = 784 

X = tf.placeholder(tf.float32, [None, n_input])

weights = {
	"encoder_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	"decoder_h1": tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}
biases = {
	"encoder_b1": tf.Variable(tf.random_normal([n_hidden_1])),
	"decoder_b1": tf.Variable(tf.random_normal([n_input])),
}

#build the encoder:
def encoder(x):
	#encoder hidden layer with sigmoid activation
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder_h1"]), 
		biases["encoder_b1"]))
	return layer_1 

#Build the decoder:
def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights["encoder_h1"])),
		biases["decoder_b1"])) 
	return layer_1 

#define the log-based function to be used in computing the KL divergence:
def log_func(x1, x2):
	return tf.multiply(x1, tf.log(tf.div(x1,x2)))

def KL_Div(rho, rho_hat):
	inv_rho = tf.subtract(tf.constant(1.), rho)
	inv_rhohat = tf.subtract(tf.constant(1.),rho_hat) 
	log_rho = log_func(rho, rho_hat) + log_func(inv_rho, inv_rhohat) 
	return log_rho 

#model definintion 
encoder_op = encoder(X) 
decoder_op = decoder(encoder_op) 
rho_hat = tf.reduce_mean(encoder_op, 1) 

#Reconstructed output 
y_pred = decoder_op 
#Targets in the input data 
y_true = X 

#Define the tensorflow ops for loss and optimization, minimize the combined 
#error Squared Reconstruction error.
cost_m = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#KL divergence regularization to introduce sparsity 
cost_sparse = 0.0001*tf.reduce_sum(KL_Div(0.2, rho_hat))
#L2 regularization of weights to keep the network stable
cost_reg = 0.0001*(tf.nn.l2_loss(weights["decoder_h1"]) + tf.nn.l2_loss(weights["encoder_h1"]))
#add up the costs 
cost = tf.add(cost_reg, tf.add(cost_m, cost_sparse)) 

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost) 

#Initializing the variables:
init = tf.global_variables_initializer() 

#Evalutation phase:
with tf.Session() as sess:
	sess.run(init) 
	total_batch = int(X_train.shape[0]/batch_size) 

	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = fetch_batch(epoch, i, batch_size) 
			_, c = sess.run([optimizer, cost], feed_dict={X:batch_xs}) 
		if epoch % display_step == 0:
			print("Epoch:", "%04d" % (epoch+1),
				"cost=", "{:.9f}".format(c))

	print("Optimization finished") 

	#Applying encoder and decode over test set 
	encode_decode = sess.run(
		y_pred, feed_dict={X:X_test[:10]})
	#Compare the original images with their reconstructions 
	f, a = plt.subplots(2, 10, figsize=(10,2))
	for i in range(10):
		a[0][i].imshow(np.reshape(X_test[i], (28,28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
	plt.show() 
	#store the decoder and encoder weights 
	dec = sess.run(weights["decoder_h1"])
	enc = sess.run(weights["encoder_h1"]) 

print(dec) 
print(enc)

##A denoising auto encoder implemenation in tensorflow:
#This can be found in page 333.
import tensorflow.contrib.layers as lays 
from skimage import transform 

#Define the netwokr with encoder and decoder 
def autoencoder(inputs):
	#encoder 
	net = lays.conv2d(inputs, 32, [5,5], stride=2, padding="SAME") 
	net = lays.conv2d(net, 16, [5,5], stride=2, padding="SAME") 
	net = lays.conv2d(net, 8, [5,5], stride=4, padding="SAME") 

	#decoder:
	net = lays.conv2d_transpose(net, 16, [5,5], stride=4, padding="SAME")
	net = lays.conv2d_transpose(net, 32, [5,5], stride=2, padding="SAME") 
	net = lays.conv2d_transpose(net, 1, [5,5], stride=2, padding="SAME", activation_fn=tf.nn.tanh) 
	return net 

def resize_batch(imgs):
	imgs = imgs.reshape((-1, 28, 28, 1))
	resize_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
	for i in range(imgs.shape[0]):
		resize_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
	return resize_imgs 

def noisy(image):
	row, col = image.shape
	mean = 0 
	var = 0.1 
	sigma = var**0.5
	gauss = np.random.normal(mean, sigma, (row,col))
	gauss = gauss.reshape(row,col)
	noisy = image + gauss 
	return noisy 

#function to define salt and pepper noise 
def s_p(image):
	row,col = image.shape 
	s_vs_p = 0.5
	amount = 0.05 
	out = np.copy(image) 
#salt mode 
	num_salt = np.ceil(amount * image.size * s_vs_p) 
	coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape] 
	out[coords] = 1 

#pepper mode 
	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape] 
	out[coords] = 0 
	return out 

#Define the ops
#Input to which the reconstructed signal is compared to 
a_e_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))
#input to the network (MNIST images) 
a_e_inputs_noise = tf.placeholder(tf.float32, (None, 32, 32, 1))
a_e_outputs = autoencoder(a_e_inputs_noise) #create autoencoder network 

#Calculate the loss and optimize the network 
loss = tf.reduce_mean(tf.square(a_e_outputs - a_e_inputs))#the mean squared error loss 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 

init = tf.global_variables_initializer() 

batch_size = 1000 
epoch_num = 10 
lr = 0.001 

#calculate the number of batches per epoch. Just like the total batches 
#command from the other computational graphs 
batch_per_ep = X_train.shape[0] // batch_size 

#evalutate the computation graph:
with tf.Session() as sess:
	sess.run(init) 
	for epoch in range(epoch_num):
		for batch_num in range(batch_per_ep):
			batch_img, batch_label = fetch_batch(epoch, batch_num, batch_size) 
			batch_img = batch_img.reshape((-1, 28, 28, 1))
			batch_img = resize_batch(batch_img) #reshapes the image sizes to 32 by 32.

			#introduce noise in the input images
			image_arr = [] 
			for i in range(len(batch_img)):
				img = batch_img[i,:,:,0] 
				img = noisy(img) 
				image_arr.append(img) 

			image_arr = np.array(image_arr) 
			image_arr = image_arr.reshape(-1, 32, 32, 1)
			_, c = sess.run([train_op, loss], feed_dict={a_e_inputs_noise:image_arr, 
				a_e_inputs: batch_img}) 
			print("epoch: {} - cost= {:.5f}".format((epoch + 1), c))

	#Test the training network 
	batch_img = X_test[:50] 
	batch_label = y_test[:50] 
	batch_img = resize_batch(batch_img) 
	image_arr = [] 

	for i in range(50):
		img = batch_img[i,:,:,0] 
		img = noisy(img) 
		image_arr.append(img) 
	image_arr = np.array(image_arr) 
	image_arr = image_arr.reshape(-1, 32, 32, 1)

	reconst_img = sess.run([a_e_outputs], feed_dict={a_e_inputs_noise: image_arr})[0]

	#plot the reconstructed images and the corresponding noisy images:
	plt.figure(1) 
	plt.title("Input Noisy Images") 
	for i in range(50):
		plt.subplot(5, 10, i+1) 
		plt.imshow(image_arr[i, ..., 0], cmap="gray") 

	plt.figure(2) 
	plt.title("Re-constructed Images") 
	for i in range(50):
		plt.subplot(5, 10, i+1) 
		plt.imshow(reconst_img[i, ..., 0], cmap="gray") 
	plt.show() 

#Important additional: The auto_encoders are trained separately once for 
#handling Gaussian noise and once for handling salt and pepper noise.









