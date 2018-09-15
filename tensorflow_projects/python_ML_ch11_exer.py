###Hands on machine learning with Tensorflow and scikit learn
###Chapter 11 treaining neural networks exercises:

##1.) According to what the lecturer (Andrew Ng) said about the subject,
#I can say that the weights for all of the neurons within each layer needs to 
#be different (even if the weighted value is initialized with the He 
#initialization technique). The reason for this is that if one was to intialize 
#an entire layer by one value, the weights between say neuron 1 in layer 2 will be 
#more or less the same as neuron 3 in layer 3. This phenominon will create a 
#neuron network that will be ineffective at picking up complex trends and as such 
#it won't scale well to the training set or to new data. 

#author's solution:
#No, all weights should be sampled independently; they should not all have the same initial value 
#One important goal of sampling weights randomly is to break symmetries:
#if all the weights have the same initial value, even if that value is not zero, then 
#symmetry is not broken (i.e. all neurons in a given layer are equivalent), and backpropagation 
#will be unable to break it. Concretely, this means that all the neurons in any 
#given layer will always have the same weights. It's like having just one neuron per layer,
#and much slower. It is viritually impossible for such a configuration to converge to a 
#good solution.

##2.)
#Author's solution:
#It is perfectly fine to intialize the bias terms to zero. Some people like to initialize 
#them just like weights, and that's okay too, it does not make much difference.

##3.) The Exponential linear unit initialization method is superior to the 
#relu initialization method because:

#First, it takes on negative values when z < 0, which allows the unit to have an average output closer to 0.
#This helps alleviate the vanishing gradients problem, as discussed easlier. The hyperparameter sigma defines 
#the value that the ELU function approaches when z is a large negative number. It is usually set to 1, but you 
#can tweak it like any other hyperparameter.

#Second, it has a nonzero gradient for z < 0 which avoids the dying units issue.

#Third, the function is smooth everywhere, including around z = 0, which helps speed up 
#gradient descent, since it does not bounce as much left and righ tof z = 0. 

##4.)
#To begin, the ELU activation function should be used if computational speed isn't really an 
#issue and the all the practioner wants to accomplish is an accurate model despite the slow computational speed 
#(which is the characteristic of ELU neural networks). Despite the slow computational speed, 
#ELU acitvation neural networks do converge faster than the other relu, tanh, and 
#sigmoid initialization methods as well as solves for the vanishing/exploding gradient 
#problems that plague the other methods.

#Leaky ReLU initialization should be used in place of ELU and normal ReLU if the practioner 
#wants to still maintain a model that has fast back-propagation speed with a little less accuracy 
#in the model's predictions. This model has the ability to solve for the vanishing gradients problem 
#(which in the case of ReLU initialized neurons is the dying ReLU problem). Even with that said though, the 
#practicioner will need to tune a different parameter (sigma, the leaking rate) and even 
#with a perfectly tuned sigma parameter some ReLU neurons can still refuse to wake up. 

#Vanilla ReLU, if want a simple neural network but don't want to deal with the pit falls of logistic 
#activation and the exploding gradients problem that accompanies the sigmoid activation function.

#The tanh activation function should be used if again the practioner doesn't want to 
#worry about the exploding gradient problem that's related to the sigmoid function curve.

#Logistic activation function should only be used as an illustration of older 
#neural network models and the pitfalls of relying on biological systems to dictate 
#efficient computational methods. The only place where the sigmoid function should be used 
#is in the output layer in the assessing of logits for a binary neural network predicting 
#only a maximum of two discrete classes.

#Softmax logistic regression should only be used as the output layers activation function 
#for the assessing of the last hidden layer's logits for classification models that are 
#predicting more than two discrete classes.

##5.) If you use the vanilla momentum optimization method with a beta argument of 1 (0.99999) 
#the gradient descent algorithm may diverge from the global minimum since the momentum coefficient 
#is increasing the learning rate by 10 times. But than again, the algorithm is set up to 
#detect that the learning rate is diverging away from the minimum and as such attempts to 
#back track down the model function. And so usually this momentum argument most likely create 
#a model that converges to the global minimum at a slower rate. 

#Author's solution: 
#If you set the momentum hyperparameter too close to 1 when using the momeuntumoptimizer(), then the 
#algorithm will likely pick up a lot of speed, hopefully roughly toward the global minimum, but than 
#it will shoot right past the minimum, due to its momentum. Then it will slow down and come back, accelerating 
#again, and so on. It may oscillate this way many times before converging so overall it will take much longer to converge 
#than with a smaller momentum value. 

##6.) 
#author's solution: 
#One way to produce a sparse model is to train the model normally, then zero out any weights. 
#for more sparsity, you can apply l_1 regularization with dual avergaging, using 
#tensorflow's FTRLOptimizer class. 

##7.) the dropout method should increase the time needed to train (or rather to 
#converge on the optimal weights for each hidden layer) since through using the algorithm 
#you're asking each neuron within the model to calculate twice the number of outputvalues. 
#With that said, I believe that prediction speed should be faster as the neurons within each 
#layer are more flexible in making predictions. 

##8.) 
#a.) to b.) 
import tensorflow as tf 
from functools import partial 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_mldata 
import matplotlib.pyplot as plt 

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

n_hidden = 100# all of the hidden layers will have a total of 100 units (hence 
#the one hidden layer number value). 
n_inputs = 28 * 28 
n_outputs = 5
learning_rate = 0.01

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
	hidden5 = tf.layers.dense(hidden4, n_hidden, kernel_initializer = he_init, 
		activation = tf.nn.elu,name = "hidden5") 
	logits = tf.layers.dense(hidden5, n_outputs, name = "outputs") 

with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
n_batches = int(np.ceil(int(X_train_4.shape[0]) / batch_size))
best_loss = np.infty
checks_without_progress = 0 

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

with tf.Session() as sess:
	saver.restore(sess, "./my_mnist_model_0_to_4.ckpt") 
	acc_test = accuracy.eval(feed_dict={X:X_test_4, y:y_test_4})
	print("Accuracy rating with Test set", acc_test) 

tf.reset_default_graph() 
#Interesting it seems that the model starts to miss classify both the training and 
#validation set after epoch 10. Now I understand why the author wants me to create an 
#early stopping model after some checkpoint. this model works perfectly, I had to look to the 
#author's solution to see how he implemented the early stopping and saving checkpoints.

#After only 25 iterations the algorithm was able to find the optimum weight values. 

#c.) the only thing that I can effectively tune is the learning rate for this neural network problem (since 
#it uses the adam optimization method). 
#author's solution:
from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.exceptions import NotFittedError 

class DNNClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, n_hidden_layers = 5, n_neurons = 100, optimizer_class = tf.train.AdamOptimizer,
				learning_rate = 0.01, batch_size = 20, activation = tf.nn.elu, initializer = he_init, 
				batch_norm_momentum = None, dropout_rate = None, random_state = None):
		"""Initialize the DNNClassifier by simply storing all the hyperparameters."""
		self.n_hidden_layers = n_hidden_layers 
		self.n_neurons = n_neurons 
		self.optimizer_class = optimizer_class 
		self.learning_rate = learning_rate 
		self.batch_size = batch_size 
		self.activation = activation 
		self.initializer = initializer 
		self.batch_norm_momentum = batch_norm_momentum 
		self.dropout_rate = dropout_rate 
		self.random_state = random_state 
		self._session = None 

	def _dnn(self, inputs):
		"""Build the hidden layers, with support for batch normalization and dropout."""
		for layer in range(self.n_hidden_layers):
			if self.dropout_rate:
				inputs = tf.layers.dropout(inputs, self.dropout_rate, training = self._training) 
			inputs = tf.layers.dense(inputs, self.n_neurons, kernel_initializer = self.initializer,
									name = "hidden%d" % (layer + 1)) 
			if self.batch_norm_momentum: 
				inputs = tf.layers.batch_normalization(inputs, momentum = self.batch_norm_momentum, 
														training = self._training) 

			inputs = self.activation(inputs, name = "hidden%d_out" % (layer + 1))
		return inputs

	def _build_graph(self, n_inputs, n_outputs):
		"""build the same model as earlier"""
		if self.random_state is not None:
			tf.set_random_seed(self.random_state) 
			np.random.seed(self.random_state) 

		X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X") 
		y = tf.placeholder(tf.int32, shape = (None), name = "y") 

		if self.batch_norm_momentum or self.dropout_rate:
			self._training = tf.placeholder_with_default(False, shape = (), name = "training") 
		else:
			self._training = None 

		dnn_outputs = self._dnn(X)

		logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer = he_init, name = "logits") 

		Y_proba = tf.nn.softmax(logits, name = "Y_proba") 

		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)

		loss = tf.reduce_mean(xentropy, name = "loss") 

		optimizer = self.optimizer_class(learning_rate = self.learning_rate) 
		training_op = optimizer.minimize(loss) 

		correct = tf.nn.in_top_k(logits, y, 1) 

		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy") 

		init = tf.global_variables_initializer() 
		saver = tf.train.Saver() 

		#make the important operations available easily through instance variables.
		self._X, self._y = X, y 
		self._Y_proba, self._loss = Y_proba, loss
		self._training_op, self._accuracy = training_op, accuracy 
		self._init, self._saver = init, saver 

	def close_session(self):
		if self._session: 
			self._session.close() 

	def _get_model_params(self):
		"""Get all variable balues (used for early stopping, faster than saving to disk)"""
		with self._graph.as_default():
			gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
		return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

	def _restore_model_params(self, model_params):
		"""Set all variables to the given values (for early stopping, faster than loading from disk)"""
		gvar_names = list(model_params.keys()) 
		assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
						for gvar_name in gvar_names}
		init_values ={gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
		feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
		self._session.run(assign_ops, feed_dict=feed_dict) 

	def fit(self, X, y, n_epochs = 100, X_valid = None, y_valid = None):
		"""
		Fit the model to the training set. If X_valid and y_valid are provided
		use early stopping
		"""

		self.close_session() 

		#Infer n_inputs and n_outputs from the training set.
		n_inputs = X.shape[1] 
		self.classes_ = np.unique(y) 
		n_outputs = len(self.classes_) 

		#Translate the labels vector to a vector of sorted class indices, 
		#containing integers from 0 to n_outputs -1 
		self.class_to_index_ = {label: index
								for index, label in enumerate(self.classes_)}
		
		y = np.array([self.class_to_index_[label]
					for label in y], dtype = np.int32) 

		self._graph = tf.Graph() 
		with self._graph.as_default():
			self._build_graph(n_inputs, n_outputs) 
			# extra ops for batch normalization 
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 

		#needed in case of early stopping 
		max_checks_without_progress = 20 
		checks_without_progress = 0 
		best_loss = np.infty 
		best_params = None 

		#Now train the model
		self._session = tf.Session(graph = self._graph)
		with self._session.as_default() as sess:
			self._init.run() 
			for epoch in range(n_epochs):
				rnd_idx = np.random.permutation(len(X))
				for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
					X_batch, y_batch = X[rnd_indices], y[rnd_indices]
					feed_dict = {self._X: X_batch, self._y:y_batch}
					if self._training is not None:
						feed_dict[self._training] = True 
					sess.run(self._training_op, feed_dict = feed_dict)
					if extra_update_ops:
						sess.run(extra_update_ops, feed_dict = feed_dict) 

				if X_valid is not None and y_valid is not None:
					loss_val, acc_val = sess.run([self._loss, self._accuracy],
												feed_dict = {self._X: X_valid,
															self._y: y_valid})

					if loss_val < best_loss:
						best_params = self._get_model_params() 
						best_loss = loss_val 
						checks_without_progress = 0 

					else: 
						checks_without_progress += 1 
					print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(epoch,
						loss_val, best_loss, acc_val * 100))

					if checks_without_progress > max_checks_without_progress:
						print("Early stopping") 
						break

				else: 
					loss_train, acc_train = sess.run([self._loss, self._accuracy], 
													feed_dict={self._X: X_batch,
																self._y: y_batch})
					print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(epoch, 
						loss_train, acc_train * 100))
			#If we used early stopping then rollback to the best model found 
			if best_params:
				self._restore_model_params(best_params) 

			return self 

	def predict_proba(self, X):
		if not self._session:
			raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__) 
		with self._session.as_default() as sess:
			return self._Y_proba.eval(feed_dict = {self._X:X}) 

	def predict(self, X):
		class_indices = np.argmax(self.predict_proba(X), axis = 1) 
		return np.array([[self.classes_[class_index]]
			for class_index in class_indices], np.int32) 

	def save(self, path):
		self._saver.save(self._session,path) 

#Now the moment of truth. I really can't believe what the author created 
#in this chapter for this problem.
from sklearn.metrics import accuracy_score

dnn_clf = DNNClassifier(random_state = 42) 
dnn_clf.fit(X_train_4, y_train_4, n_epochs = 1000, X_valid = X_valid_4, y_valid = y_valid_4)
y_pred = dnn_clf.predict(X_test_4) 
print(accuracy_score(y_test_4, y_pred))# The accuracy rating between the two methods (the class program and the vanilla 
#tensorflow session are some what similar). I believe that that variability is minimal at best.

#the DNNClassifier class that the author create is amazing but I can see a significant 
#speed drop between using this class and vanilla method of just simply using the tensorflow session 
#without any class or function wrappers. This must be the pit falls of designing your own applications 
#with someones framework. 
from sklearn.model_selection import RandomizedSearchCV

def leaky_relu(alpha = 0.01):
	def parametrized_leaky_relu(z, name = None):
		return tf.maximum(alpha * z, z, name = name) 
	return parametrized_leaky_relu 

param_grid = {"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
"n_neurons": [10, 30, 50, 70, 90, 100, 120, 150],
"learning_rate": [0.01, 0.02, 0.05, 0.1],
 "activation": [tf.nn.elu, tf.nn.relu, leaky_relu(alpha = 0.01), leaky_relu(alpha = 0.1)]
 }
fit_params = {"X_valid": X_valid_4, "y_valid": y_valid_4, "n_epochs": 1000}

#rnd_search = RandomizedSearchCV(DNNClassifier(random_state = 42), param_grid, n_iter = 50, fit_params=fit_params, random_state = 42, verbose =2)

#rnd_search.fit(X_train_4, y_train_4)
#I was forced this grid search session because the computation was taking too long 
#for my computer to handle. will need to come back to this problem once I have a better system 
#(or rather more nodes to process large computations).


#d.) Adding batch normalization to the resulting model: 
from functools import partial

tf.reset_default_graph()

learning_rate = 0.01 
n_inputs = 28 * 28 
n_outputs = 5
n_hidden = 90 

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X") 
y = tf.placeholder(tf.int32, shape = (None), name = "y")
training = tf.placeholder_with_default(False, shape = (), name = "training")  

my_batch_norm_layer = partial(tf.layers.batch_normalization, training = training, 
							momentum = 0.95)

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X, n_hidden, name = "hidden1") 
	bn1 = my_batch_norm_layer(hidden1) 
	bn1_act = tf.nn.leaky_relu(bn1, alpha = 0.01) 
	hidden2 = tf.layers.dense(bn1_act, n_hidden, name = "hidden2") 
	bn2 = my_batch_norm_layer(hidden2) 
	bn2_act = tf.nn.leaky_relu(bn2, alpha = 0.01) 
	hidden3 = tf.layers.dense(bn2_act, n_hidden, name = "hidden3") 
	bn3 = my_batch_norm_layer(hidden3) 
	bn3_act = tf.nn.leaky_relu(bn3, alpha = 0.01) 
	hidden4 = tf.layers.dense(bn3_act, n_hidden, name = "hidden4") 
	bn4 = my_batch_norm_layer(hidden4) 
	bn4_act = tf.nn.leaky_relu(bn4, alpha = 0.01) 
	logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = "outputs") 
	logits = my_batch_norm_layer(logits_before_bn) 


with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
n_batches = int(np.ceil(int(X_train_4.shape[0]) / batch_size))
best_loss = np.infty
checks_without_progress = 0 

init = tf.global_variables_initializer()
saver = tf.train.Saver()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
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

with tf.Session() as sess:
	saver.restore(sess, "./my_mnist_model_0_to_4.ckpt") 
	acc_test = accuracy.eval(feed_dict={X:X_test_4, y:y_test_4})
	print("Accuracy rating with Test set", acc_test)

tf.reset_default_graph()# This model has obtained an accuracy rating of about 
#0.9898 on the testing set which is very good compared to the other models. The main problems with 
#this implementation is the problem of the model overfitting the training data (which seems to be 
#the case for the training accuracy rate is almost 100 percent for most of the training 
#process)   

#e.) Adding dropout to every hidden layer with the model (this might conflict 
#with the batch normalization method). Will need to see if this is the case.
from functools import partial

tf.reset_default_graph()

learning_rate = 0.01 
n_inputs = 28 * 28 
n_outputs = 5
n_hidden = 90 
dropout_rate = 0.5

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X") 
y = tf.placeholder(tf.int32, shape = (None), name = "y")
training = tf.placeholder_with_default(False, shape = (), name = "training")  

X_drop = tf.layers.dropout(X, dropout_rate, training = training) 
my_batch_norm_layer = partial(tf.layers.batch_normalization, training = training, 
							momentum = 0.95)

with tf.name_scope("dnn"):
	hidden1 = tf.layers.dense(X_drop, n_hidden, name = "hidden1")
	hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training = training)  
	bn1 = my_batch_norm_layer(hidden1_drop) 
	bn1_act = tf.nn.leaky_relu(bn1, alpha = 0.01) 
	hidden2 = tf.layers.dense(bn1_act, n_hidden, name = "hidden2")
	hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training = training)
	bn2 = my_batch_norm_layer(hidden2_drop) 
	bn2_act = tf.nn.leaky_relu(bn2, alpha = 0.01) 
	hidden3 = tf.layers.dense(bn2_act, n_hidden, name = "hidden3") 
	hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training = training)
	bn3 = my_batch_norm_layer(hidden3_drop) 
	bn3_act = tf.nn.leaky_relu(bn3, alpha = 0.01) 
	hidden4 = tf.layers.dense(bn3_act, n_hidden, name = "hidden4")
	hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training = training)
	bn4 = my_batch_norm_layer(hidden4_drop) 
	bn4_act = tf.nn.leaky_relu(bn4, alpha = 0.01) 
	logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = "outputs") 
	logits = my_batch_norm_layer(logits_before_bn) 


with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) 
	loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
n_batches = int(np.ceil(int(X_train_4.shape[0]) / batch_size))
best_loss = np.infty
checks_without_progress = 0 

init = tf.global_variables_initializer()
saver = tf.train.Saver()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  

with tf.Session() as sess:
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
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

with tf.Session() as sess:
	saver.restore(sess, "./my_mnist_model_0_to_4.ckpt") 
	acc_test = accuracy.eval(feed_dict={X:X_test_4, y:y_test_4})
	print("Accuracy rating with Test set", acc_test)
#this model obtained an accuracy rating of about 99 percent with only a small reduction in 
#training set accuracy (which means that the model with just a leaky relu activation function 
#and batch normalization is more than sufficient). Interestingly using the drop out 
#method increased convergence time. Will need to look into if this is a characteristic of this 
#technique. 

tf.reset_default_graph()







