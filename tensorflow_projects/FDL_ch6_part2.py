###Fundamentals of deep learning chapter 6 part 2:
###Visualizing the components within a hidden layer. 

import numpy as np 
import tensorflow as tf
from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 
from tensorflow.python import control_flow_ops
from sklearn import decomposition
import matplotlib.pyplot as plt     
import FDL_ch6_part1 as ae 
import argparse

def read_infile():
	n_classes = 10
	mnist = fetch_mldata("MNIST original") 
	X_mnist, y_mnist = mnist["data"], mnist["target"]
	X_mnist = X_mnist.astype(np.float32)/255
	y_mnist = y_mnist.astype(np.int32)  
	X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)    
	return X_train, y_train, X_test, y_test 

X_train, y_train, X_test, y_test = read_infile() 

def scatter(codes, labels):
	colors = [
		('#27ae60', 'o'),
		('#2980b9', 'o'),
		('#8e44ad', 'o'),
		('#f39c12', 'o'),
		('#c0392b', 'o'),
		("#27ae60", 'x'),
		('#2980b9', 'x'),
		('#8e44ad', 'x'),
		('#c0392b', 'x'),
		('#f39c12', 'x'),
	]
	for num in range(10):
		plt.scatter([codes[:0][i] for i in range(len(labels)) if labels[i] == num],
			[codes[:,1][i] for i in range(len(labels)) if labels[i] == num], 7,
			label = str(num), color = colors[num][0], marker=colors[num][1])

	plt.legend() 
	plt.show() 

with tf.Graph().as_default():
	with tf.variable_scope("autoencoder_model"): 
		x = tf.placeholder(tf.float32, [None, 784])
		phase_train=tf.placeholder(tf.bool)

		code = ae.encoder(x, 2, phase_train) 

		output = ae.decoder(code, 2, phase_train) 

		cost, train_summary_op = ae.loss(output, x) 

		global_step = tf.Variable(0, name="global_step", trainable=False) 

		train_op = ae.training(cost, global_step) 

		eval_op, in_im_op, out_im_op, val_summary_op = ae.evaluate(output, x) 

		sess = tf.Session() 
		saver = tf.train.Saver() 
		saver.restore(sess, args.savepath[0]) 

		ae_codes = sess.run(code, feed_dict={x:X_test, train_phase:True})

		scatter(ae_codes, y_test) 
		

##PCA linear encoding of the mnist dataset:

pca = decomposition.PCA(n_components=2) 
pca.fit(X_train) 
pca_codes = pca.transform(X_test)

pca_recon = pca.inverse_transform(pca_codes[:5]) 
plt.imshow(pca_recon[1].reshape((28,28)), cmap=plt.cm.gray) 
plt.show()

scatter(pca_codes, y_test)
#Cool this code implementation is actually working. Well kind of 
#the code is actually attempting to retrain itself. I believe that the 
#implemenation within the book hands on machine learning using the .ckpt 
#data format is really the best way to reuse past tensorflow models. 

##Diagnosing an autoencoder through corrupting the input data using the 
##Mnist dataset yet again.
def corrupt_input(x):
	corrupting_matrix = tf.random_uniform(shape=tf.shape(x), minval=0, 
		maxval=2, dtype=tf.int32) 

	return x * tf.cast(corrupting_matrix, tf.float32) 

corrupt = tf.placeholder(tf.float32, [None, 784]) 
phase_train = tf.placeholder(tf.bool) 
c_x = (corrupt_input(x) * corrupt) + (x * (1- corrupt)) 
#Then you ultimately place the c_x object into the autoencoder 
#computational graph. 










