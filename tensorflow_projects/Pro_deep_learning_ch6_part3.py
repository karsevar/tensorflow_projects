###Pro deep learning chapter 6 
##Creating a Generative Adverserial Neural network using Nash equalibrium 
#and the zero sum game concept.
import tensorflow as tf 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import numpy as np   
from sklearn.datasets import fetch_mldata

def read_infile():
	n_classes = 10
	mnist = fetch_mldata("MNIST original") 
	X_mnist, y_mnist = mnist["data"], mnist["target"]
	X_mnist = X_mnist.astype(np.float32).reshape(-1, 28 * 28)/255#Additional step since I'm assuming that 
	#the tanh function is a distribution of values between -1 and 1 it is only understandable to normalize 
	#the mnist dataset values by the same distribution.
	y_mnist = np.eye(n_classes)[y_mnist.astype(np.int32)]  
	X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)    
	return X_train, y_train, X_test, y_test 

X_train, y_train, X_test, y_test = read_infile()
print(len(X_train))  
#The dimension of the Proir Noise signal is taken to be 100 
#The generator would have 150 and 300 hidden units successively before 784 outputs 
#corresponding to 28 by 28 image size.

h1_dim = 150 
h2_dim = 300 
dim = 100 
batch_size = 256 

#Define the generator - take noise and convert them to images:
def generator_(z_noise):
	w1 = tf.Variable(tf.truncated_normal([dim, h1_dim], stddev=0.1), name="w1_g", 
		dtype=tf.float32) 
	b1 = tf.Variable(tf.zeros([h1_dim]), name="b1_g", dtype=tf.float32) 
	h1 = tf.nn.relu(tf.matmul(z_noise, w1) + b1) 
	w2 = tf.Variable(tf.truncated_normal([h1_dim, h2_dim], stddev=0.1), name="w2_g", 
		dtype=tf.float32) 
	b2 = tf.Variable(tf.zeros([h2_dim]), name="b2_g", dtype=tf.float32)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2) 
	w3 = tf.Variable(tf.truncated_normal([h2_dim, 28*28], stddev=0.1), name="w3_g", dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([28*28]), name="b3_g", dtype=tf.float32) 
	h3 = tf.matmul(h2, w3) + b3
	out_gen = tf.nn.relu(h3) 
	weights_g = [w1, b1, w2, b2, w3, b3] 
	return out_gen, weights_g 

#Define the discriminator:
def discriminator_(x, out_gen, keep_prob):
	x_all = tf.concat([x, out_gen], 0) 
	w1 = tf.Variable(tf.truncated_normal([28*28, h2_dim], stddev=0.1), name="w1_d", 
		dtype=tf.float32) 
	b1 = tf.Variable(tf.zeros([h2_dim]), name="b1_d", dtype=tf.float32) 
	h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_all, w1) + b1), keep_prob) 
	w2 = tf.Variable(tf.truncated_normal([h2_dim, h1_dim], stddev=0.1), name="w2_d") 
	b2 = tf.Variable(tf.zeros([h1_dim]), name="b2_d", dtype=tf.float32) 
	h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob) 
	w3 = tf.Variable(tf.truncated_normal([h1_dim, 1], stddev=0.1), name="w3_d", dtype=tf.float32) 
	b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32) 
	h3 = tf.matmul(h2, w3) + b3
	y_data = tf.nn.sigmoid(tf.slice(h3, [0,0], [batch_size, -1], name=None)) 
	y_fake = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1,-1], name=None)) 
	weights_d = [w1, b1, w2, b2, w3, b3] 
	return y_data, y_fake, weights_d 

x = tf.placeholder(tf.float32, [batch_size, 28*28], name="x_data") 
z_noise = tf.placeholder(tf.float32, [batch_size, dim], name="z_prior")
#Dropout probability 
keep_prob = tf.placeholder(tf.float32, name="keep_prob") 
#generate the output ops for generator and also define the weights.
out_gen, weights_g = generator_(z_noise) 
#define the ops and weights of disciminator 
y_data, y_fake, weights_d = discriminator_(x, out_gen, keep_prob) 

def fetch_batch(batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

#Cost function for discriminator and generator:
discr_loss = -1 * tf.reduce_mean(tf.log(y_data) + tf.log(1 - y_fake)) 
gen_loss = -1 * tf.reduce_mean(tf.log(y_fake)) 
optimizer = tf.train.AdamOptimizer(0.0001) 
d_trainer = optimizer.minimize(discr_loss, var_list=weights_d) 
g_trainer = optimizer.minimize(gen_loss, var_list=weights_g) 
init = tf.global_variables_initializer() 
saver = tf.train.Saver() 

with tf.Session() as sess: 
	sess.run(init) 
	z_sample = np.random.uniform(-1, 1, size=(batch_size, dim)).astype(np.float32) 

	for i in range(3000):
		batch_x, _ = fetch_batch(i, batch_size) 
		x_value = 2*batch_x.astype(np.float32) - 1 
		z_value = np.random.uniform(-1, 1, size=(batch_size, dim)).astype(np.float32) 
		sess.run(d_trainer, feed_dict={x:x_value, z_noise:z_value, keep_prob:0.50})
		sess.run(g_trainer, feed_dict={x:x_value, z_noise:z_value, keep_prob:0.50})
		c1, c2 = sess.run([discr_loss, gen_loss], feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7}) 
		print("iter:", i, "cost of discriminator", c1, "cost of generator", c2) 

	out_val_img = sess.run(out_gen, feed_dict={z_noise:z_sample}) 

	imgs = 0.5*(out_val_img + 1) 
	for k in range(36):
		plt.subplot(6,6,k+1) 
		image = np.reshape(imgs[k], (28,28)) 
		plt.imshow(image, cmap="gray") 
	plt.show()
#I think that this model is suffering from the vanishing gradient problem. The cost function of the 
#generator is expoding and the cost function of the discriminator is vanishing. Will need to 
#see why this is.

