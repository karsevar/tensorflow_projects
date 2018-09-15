###Fundametals of Deep learning chapter 6 part 1 
###Implementing an autoencoder in tensorflow:
#the following code is based off of the seminal experiments done in 
#the paper Reducing the dimensionality of data with neural networks.

import numpy as np 
import tensorflow as tf
from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 
from tensorflow.python import control_flow_ops  

learning_rate = 0.01 
training_epochs = 40 
batch_size = 100 
display_step = 1 

n_encoder_hidden_1 = 1000
n_encoder_hidden_2 = 500 
n_encoder_hidden_3 = 250 
n_decoder_hidden_1 = 250 
n_decoder_hidden_2 = 500 
n_decoder_hidden_3 = 1000 
n_code = 5

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 
 
def read_infile():
	n_classes = 10
	mnist = fetch_mldata("MNIST original") 
	X_mnist, y_mnist = mnist["data"], mnist["target"]
	X_mnist = X_mnist.astype(np.float32)/255
	y_mnist = np.eye(n_classes)[y_mnist.astype(np.int32)]  
	X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)    
	return X_train, y_train, X_test, y_test 

X_train, y_train, X_test, y_test = read_infile()

X_val, y_val = X_train[:6000], y_train[:6000]

def layer_batch_norm(X, n_out, phase_train):
	beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32) 
	gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32) 

	beta = tf.get_variable("beta", [n_out], initializer=beta_init) 
	gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init) 

	batch_mean, batch_var = tf.nn.moments(X, [0], name="moments")
	ema = tf.train.ExponentialMovingAverage(decay=0.9) 
	ema_apply_op = ema.apply([batch_mean, batch_var]) 
	ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var) 

	def mean_var_with_update():
		with tf.control_dependencies([ema_apply_op]):
			return tf.identity(batch_mean), tf.identity(batch_var) 

	mean, var = control_flow_ops.cond( phase_train, mean_var_with_update, lambda: (ema_mean, ema_var)) 

	reshaped_x = tf.reshape(X, [-1, 1, 1, n_out]) 
	normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var, beta, gamma, 1e-3, True) 
	return tf.reshape(normed, [-1, n_out]) 


def encoder(X, n_code, phase_train):
	with tf.variable_scope("encoder"):
		with tf.variable_scope("hidden_1"):
			hidden_1 = layer(X, [784, n_encoder_hidden_1],
				[n_encoder_hidden_1], phase_train)

		with tf.variable_scope("hidden_2"):
			hidden_2 = layer(hidden_1, [n_encoder_hidden_1, n_encoder_hidden_2], 
				[n_encoder_hidden_2], phase_train) 

		with tf.variable_scope("hidden_3"):
			hidden_3 = layer(hidden_2, [n_encoder_hidden_2, n_encoder_hidden_3], 
				[n_encoder_hidden_3], phase_train) 

		with tf.variable_scope("code"):
			code = layer(hidden_3, [n_encoder_hidden_3, n_code], [n_code], 
				phase_train)
	return code  	  

def decoder(code, n_code, phase_train):
	with tf.variable_scope("decoder"):
		with tf.variable_scope("hidden_1"):
			hidden_1 = layer(code, [n_code, n_decoder_hidden_1],
				[n_decoder_hidden_1], phase_train) 

		with tf.variable_scope("hidden_2"):
			hidden_2 = layer(hidden_1, [n_decoder_hidden_1, n_decoder_hidden_2], 
				[n_decoder_hidden_2], phase_train)

		with tf.variable_scope("hidden_3"):
			hidden_3 = layer(hidden_2, [n_decoder_hidden_2, n_decoder_hidden_3],
				[n_decoder_hidden_3], phase_train) 

		with tf.variable_scope("output"):
			output = layer(hidden_3, [n_decoder_hidden_3, 784], 
				[784], phase_train)

		return output  

def layer(input, weight_shape, bias_shape, phase_train):
	weight_stddev = (1.0/weight_shape[0]) ** 0.5
	weight_init = tf.random_normal_initializer(stddev=weight_stddev) 
	bias_init = tf.constant_initializer(value=0) 
	W = tf.get_variable("W", weight_shape, initializer=weight_init)
	b = tf.get_variable("b", bias_shape, initializer=bias_init) 
	logits = tf.matmul(input, W) + b
	return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))

#the cost function with is used. The L2 distance between the original image the 
#the reconstructed image.
#Interesting addition to this code is the use of scalar_summary to log 
#the error incurrec at every minibatch

def loss(output, X):
	with tf.variable_scope("training"):
		l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, X)), 1))
		train_loss = tf.reduce_mean(l2) 
		train_summary_op = tf.summary.scalar("train_cost", train_loss) 
		return train_loss, train_summary_op 

def training(cost, global_step):
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001, 
		beta1 = 0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name="adam") 
	train_op = optimizer.minimize(cost, global_step=global_step) 
	return train_op 

#Additional functionality. As usual we'll use a validation dataset and compute the same L2 
#norm measurement for model evaluation. In addition, we'll collect image summaries so that we can compare 
#both the input images and reconstructions.
def image_summary(summary_label, tensor):
	tensor_reshaped = tf.reshape(tensor, [-1, 28, 28, 1]) 
	return tf.summary.image(summary_label, tensor_reshaped) 

def evaluate(output, X):
	with tf.variable_scope("validation"):
		in_im_op = image_summary("input_image", X)
		out_im_op = image_summary("output_image", output) 
		l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, X, name="val_diff")), 1))
		val_loss = tf.reduce_mean(l2)
		val_summary_op = tf.summary.scalar("val_cost", val_loss) 
		return val_loss, in_im_op, out_im_op, val_summary_op 

with tf.variable_scope("autoencoder_model"):
	X = tf.placeholder(tf.float32, [None, 784]) 
	phase_train = tf.placeholder(tf.bool) 
	code = encoder(X, int(n_code), phase_train) 
	output = decoder(code, int(n_code), phase_train) 
	cost, train_summary_op = loss(output, X) 
	global_step = tf.Variable(0, name="global_step", trainable=False)
	train_op = training(cost, global_step) 
	eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, X) 
	summary_op = tf.summary.merge_all() 

	saver = tf.train.Saver(max_to_keep=200) 

	sess = tf.Session()

	init = tf.global_variables_initializer() 

	sess.run(init) 

	#Training cycle 
	for epoch in range(training_epochs):
		 avg_cost = 0 
		 total_batch = int(X_train.shape[0]/batch_size) 
		 for batch_index in range(total_batch):
		 	batch_X, batch_y = fetch_batch(epoch, batch_index, batch_size) 
		 	_, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={X: batch_X, phase_train: True}) 

		 	avg_cost += new_cost/total_batch

		 if epoch % display_step == 0:
		 	print("Epoch:", "%04d" % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

		 	validation_loss, in_im, out_im, val_summary = sess.run([eval_op, in_im_op, out_im_op, val_summary_op], feed_dict={X: X_val, phase_train: True}) 
		 	print("Validation Loss:", validation_loss) 

		 	saver.save(sess, "mnist_autoencoder_hidden=" + str(n_code) + "_logs/model-checkpoint-" + "%04d" % (epoch+1), global_step=global_step) 

	print("optimization Finished") 

	test_loss = sess.run(eval_op, feed_dict={X:X_test, phase_train: False}) 
	print("Test Loss:", test_loss) 

#This autoencoder model isn't really working that well in my machine will need to look into the 
#other implementations created in hands on machine learning and pro deep learning to see what I did 
#wrong and the implemenations that could have been batcher in the initial code implementation. 






