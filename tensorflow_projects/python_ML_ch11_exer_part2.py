###Hands on machine learning with sklearn and tensorflow 
###Chapter 11 exercises part 2:

##9.)
#a.) through c.) kind of (since I didn't split the training data into groups 
#of 100).  
import tensorflow as tf 
from functools import partial 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_mldata 

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]
X_mnist_04, y_mnist_04 = (y_mnist <= 4), (y_mnist <= 4) 

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_mnist[X_mnist_04], y_mnist[y_mnist_04], test_size=0.2, 
															random_state = 42, shuffle = True)

X_train_4 = X_train_4.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test_4 = X_test_4.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train_4 = y_train_4.astype(np.int32) 
y_test_4 = y_test_4.astype(np.int32)
X_valid_4, X_train_4 = X_train_4[:5000], X_train_4[5000:] 
y_valid_4, y_train_4 = y_train_4[:5000], y_train_4[5000:]

X_mnist_09, y_mnist_09 = (y_mnist >= 5), (y_mnist >= 5)
X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(X_mnist[X_mnist_09], y_mnist[y_mnist_09], test_size=0.2, 
	random_state = 42, shuffle = True)
X_train_9 = X_train_9.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test_9 = X_test_9.astype(np.float32).reshape(-1, 28*28) / 255.0 
y_train_9 = y_train_9.astype(np.int32) 
y_test_9 = y_test_9.astype(np.int32)
X_valid_9, X_train_9 = X_train_9[:5000], X_train_9[5000:] 
y_valid_9, y_train_9 = y_train_9[:5000], y_train_9[5000:] 

learning_rate = 0.01 
n_inputs = 28 * 28 
n_outputs = 5
n_hidden = 90 
dropout_rate = 0.5

he_init = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X") 
y = tf.placeholder(tf.int32, shape = (None), name = "y")    

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden, name = "hidden1") 
	hidden2 = tf.layers.dense(hidden1, n_hidden, kernel_initializer = he_init, 
		activation = tf.nn.elu, name = "hidden2") 
	hidden3 = tf.layers.dense(hidden2, n_hidden, kernel_initializer = he_init, 
		activation = tf.nn.elu, name = "hidden3") 
	hidden4 = tf.layers.dense(hidden3, n_hidden, kernel_initializer = he_init, 
		activation = tf.nn.elu,name = "hidden4") 
	logits = tf.layers.dense(hidden4, n_outputs, name = "outputs") 
	
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")
	Y_proba = tf.nn.softmax(logits, name = "Y_proba") 

with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 
	training_op = optimizer.minimize(loss)


def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train_4), size = batch_size) 
	X_batch = X_train_4[indices] 
	y_batch = y_train_4[indices] 
	return X_batch, y_batch 

n_epochs = 100 
batch_size = 100
n_batches = int(np.ceil(int(X_train_9.shape[0]) / batch_size))
best_loss = np.infty
checks_without_progress = 0 
max_checks_without_progress = 20  

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		accuracy_val = accuracy.eval(feed_dict={X: X_valid_4, y: y_valid_4})
		accuracy_train = accuracy.eval(feed_dict={X: X_train_4, y: y_train_4})
		loss_val = sess.run(loss, feed_dict={X:X_valid_4, y:y_valid_4}) 
		if loss_val < best_loss:
			save_path = saver.save(sess, "./my_mnist_model_0_to_4.ckpt") 
			best_loss = loss_val 
			checks_without_progress = 0 
		else: 
			checks_without_progress += 1 
			if checks_without_progress > 20:
				print("Early_stopping") 
				break
		print(epoch, "loss val:", loss_val, "best loss:", best_loss, "training accuracy:", accuracy_train, "Validation accuracy:", accuracy_val)  

tf.reset_default_graph() 

#Update, it's important to keep in mind that the author desided to use the 
#regular vanilla neural network framework without the fancy batch normalization,
#leaky relu and neuron dropout methods since the inclusion of such method will 
#complicate the gradient freezing process and the inclusion of the three mentioned 
#methods will only increase the accuracy by only 0.3 percent. 

#a.) author's solution:
restore_saver = tf.train.import_meta_graph("./my_mnist_model_0_to_4.ckpt.meta") 

X = tf.get_default_graph().get_tensor_by_name("X:0") 
y = tf.get_default_graph().get_tensor_by_name("y:0") 
loss = tf.get_default_graph().get_tensor_by_name("loss/loss:0") 
Y_proba = tf.get_default_graph().get_tensor_by_name("eval/Y_proba:0") 
logits = Y_proba.op.inputs[0] 
accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0") 

learning_rate = 0.01 

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "hidden[1234]")  
optimizer = tf.train.AdamOptimizer(learning_rate, name = "Adam2") 
training_op = optimizer.minimize(loss, var_list = output_layer_vars) 

correct = tf.nn.in_top_k(logits, y, 1) 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

init = tf.global_variables_initializer() 
four_frozen_saver = tf.train.Saver()

X_mnist_09, y_mnist_09 = X_mnist[y_mnist >= 5], y_mnist[y_mnist >= 5] - 5 
#Cheater, the author just changed the output values from 5 to 9 to 0 to 4. Ingenius and depressing at the sametime 
#since I'm just wonder if there is a way to train a different output layer from a pretrained model. 

X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(X_mnist_09, y_mnist_09, test_size=0.2, 
	random_state = 42, shuffle = True)
X_train_9 = X_train_9.astype(np.float32).reshape(-1, 28*28)/255.0 
X_test_9 = X_test_9.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train_9 = y_train_9.astype(np.int32) 
y_test_9 = y_test_9.astype(np.int32) 
X_valid_9, X_train_9 = X_train_9[:5000], X_train_9[5000:] 
y_valid_9, y_train_9 = y_train_9[:5000], y_train_9[5000:]

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train_9), size = batch_size) 
	X_batch = X_train_9[indices] 
	y_batch = y_train_9[indices] 
	return X_batch, y_batch   

import time 

n_epochs = 100 
batch_size = 100
n_batches = int(np.ceil(int(X_train_9.shape[0]) / batch_size))
best_loss = np.infty
checks_without_progress = 0 
max_checks_without_progress = 20  

with tf.Session() as sess:
	init.run() 
	restore_saver.restore(sess, "./my_mnist_model_0_to_4.ckpt") 
	for var in output_layer_vars:
		var.initializer.run() 

	t0 = time.time() 

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		accuracy_val = accuracy.eval(feed_dict={X: X_valid_9, y: y_valid_9})
		accuracy_train = accuracy.eval(feed_dict={X: X_train_9, y: y_train_9})
		loss_val = sess.run(loss, feed_dict={X:X_valid_9, y:y_valid_9}) 
		if loss_val < best_loss:
			save_path = four_frozen_saver.save(sess, "./my_mnist_model_5_to_9_four_frozen.ckpt") 
			best_loss = loss_val 
			checks_without_progress = 0 
		else: 
			checks_without_progress += 1 
			if checks_without_progress > 20:
				print("Early_stopping") 
				break
		print(epoch, "loss val:", loss_val, "best loss:", best_loss, "training accuracy:", accuracy_train, "Validation accuracy:", accuracy_val)  

	t1 = time.time() 
	print("total training time: {:.1f}s".format(t1 - t0)) 

with tf.Session() as sess:
	four_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_four_frozen.ckpt") 
	acc_test = accuracy.eval(feed_dict={X: X_test_9, y: y_test_9}) 
	print("Final test accuracy: {:.2f}%".format(acc_test * 100)) 














