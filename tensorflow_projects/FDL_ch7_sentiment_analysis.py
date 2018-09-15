###Fundamentals of deep learning chapter 7:
##Implementing a sentiment analysis:
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np 
import tensorflow as tf  

training_epochs = 1000 
batch_size = 32 
display_step = 1 

train, test, _ = imdb.load_data(path="data/imdb.pkl", n_words = 30000, 
	valid_portion=0.1) 
   
#I think I now understand why I couldn't load the mnist dataset 
#onto my console. Most likely the load_data(path=) command is asking 
#where I would like to download the dataset into. Will need to test this 
#out. 

trainX, trainY = train 
testX, testY = test 

#print(trainX[:50])#So from what I can see all of the words have already been encoded into 
#word vector tags (where each word has a unique numerical value most likely illustrating it's 
#place in the corpus). Will need to see how this was created and how can one decode these values 
#for exploritory analysis.  
trainX = pad_sequences(trainX, maxlen=500, value=0.) 
testX = pad_sequences(testX, maxlen=500, value=0.)
#The idea for this line is the pad the reviews so that the 
#entries will have a uniform word length of 500 words. Interesting that the 
#author didn't use a network architecture that could take varying sequence lengths.

trainY = to_categorical(trainY, nb_classes=2) 
testY = to_categorical(testY, nb_classes=2) 

#The following is a class that will be used to serve minibatches 
#from the underlying dataset. 

class IMDBDataset(): 
	def __init__(self, X, Y):
		self.num_examples = len(X)
		self.inputs = X
		self.tags = Y 
		self.ptr = 0 

	def minibatch(self, size): 
		ret = None 
		if self.ptr + size < len(self.inputs): 
			ret = self.inputs[self.ptr:self.ptr+size], self.tags[self.ptr:self.ptr+size] 
		else: 
			ret = np.concatenate((self.inputs[self.ptr:], self.inputs[:size-len(self.inputs[self.ptr:])])), np.concatenate((self.tags[self.ptr:], self.tags[:size-len(self.tags[self.ptr:])]))
		self.ptr = (self.ptr + size) % len(self.inputs) 

		return ret 

train = IMDBDataset(trainX, trainY) 
val = IMDBDataset(testX, testY) 

#the embedding layer used to translate the embedded word vectors into actual 
#linguistic characters. 
def embedding_layer(input, weight_shape): 
	weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0]) **0.5)
	E = tf.get_variable("E", weight_shape, initializer=weight_init) 
	incoming = tf.cast(input, tf.int32) 
	embeddings = tf.nn.embedding_lookup(E, incoming) 
	return embeddings 

#Interesting note, We do some extra work to pull out the last output emitted by the 
#LSTM using the tf.slice and tf.squeeze operators, which find the extract slice that contains 
#the last output of the LSTM and then eliminates the unnecessary dimension 

#Implemenation of the LSTM: 
def lstm(input, hidden_dim, keep_prob, phase_train): 
	lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
	dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob) 
	#It seems that in place of using gradient clipping as the normalization and anti overfitting method  
	#the author is using cell dropout using the dropoutwrapper function. 
	#stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * 2, state_is_tuple=True) 
	lstm_outputs, state = tf.nn.dynamic_rnn(dropout_lstm, input, dtype=tf.float32) 
	return tf.reduce_max(lstm_outputs, reduction_indices=[1])  

#Batch normalization layer. I hope this function doesn't have any errors:
def layer_batch_norm(x, n_out, phase_train): 
	beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32) 
	gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32) 

	beta = tf.get_variable("beta", [n_out], initializer=beta_init) 
	gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init) 

	batch_mean, batch_var = tf.nn.moments(x, [0], name="moments")
	ema = tf.train.ExponentialMovingAverage(decay=0.9) 
	ema_apply_op = ema.apply([batch_mean, batch_var])
	ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var) 
	def mean_var_with_update():
		with tf.control_dependencies([ema_apply_op]):
			return tf.identity(batch_mean), tf.identity(batch_var) 

	mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var)) 

	reshaped_x = tf.reshape(x, [-1, 1, 1, n_out]) 
	normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var, beta, 
		gamma, 1e-3, True) 
	return tf.reshape(normed, [-1, n_out]) 

#The following function creates the recurrent neural network with the different 
#neurons. 
def layer(input, weight_shape, bias_shape, phase_train):
	weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5) 
	bias_init = tf.constant_initializer(value=0) 
	W = tf.get_variable("W", weight_shape, initializer=weight_init) 
	b = tf.get_variable("b", bias_shape, initializer=bias_init) 
	logits = tf.matmul(input, W) + b
	return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train)) 

#The following function puts all of the components together in a large overall 
#computational graph. 
def inference(input, phase_train):
	embedding = embedding_layer(input, [30000, 512]) 
	lstm_output = lstm(embedding, 512, 0.5, phase_train) 
	output = layer(lstm_output, [512, 2], [2], phase_train) 
	return output 

#This creates the cost function and the partial derivatives for the backpropagation 
#through time algorithm: 
def loss(output, y):
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)#No need for this 
	#cost function equatin since the classification problem is only binary. And so 
	#log loss can be used instead. 
	loss = tf.reduce_mean(xentropy) 
	return loss

def training(cost, global_step):
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name="Adam") 
	train_op = optimizer.minimize(cost, global_step=global_step) 

def evaluate(output, y):
	correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
	return accuracy

x = tf.placeholder(tf.float32, [None, 500]) 
y = tf.placeholder(tf.float32, [None, 2]) 
phase_train = tf.placeholder(tf.bool)#Since dropout can only be used for 
#training the algorithm and not for testing the model. 
me = [] 
output = inference(x, phase_train) 
cost = loss(output, y)
global_step = tf.Variable(0, name="global_step", trainable=False)  
train_op = training(cost, global_step) 
eval_op = evaluate(output, y)

x, y = train.minibatch(batch_size) 
print(x)  
print(y)    

#Evaluation phase of the graph: 
init = tf.global_variables_initializer() 

with tf.Session() as sess: 
	sess.run(init) 

	for epoch in range(training_epochs):
		avg_cost = 0. 
		total_batch = int(train.num_examples/batch_size) 
		print("total of %d minibatches in epoch %d" % (total_batch,epoch)) 

		#loop over all batches:
		for i in range(total_batch):
			minibatch_x, minibatch_y = train.minibatch(batch_size) 
			_, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x, y:minibatch_y, phase_train:True}) 
			avg_cost += new_cost/total_batch 
			print("Training cost for batch %d in epoch %d was:" % (i, epoch), new_cost) 
			if i % 100 == 0:
				print("Epoch:", "%04d" % (epoch+1), "Minibatch:", "%04d" % (i+1), "cost =", "{:.9f}".format((avg_cost * total_batch)/(i+1)))
				val_x, val_y = val.minibatch(val.num_examples) 
				val_accuracy = sess.run(eval_op, feed_dict={x:val_x, y:val_y, train_phase:False}) 
				print("Validation Accuracy:", val_accuracy) 

	print("Opimization Finished") 

#Other model that doesn't seem to work with this book. Will need to come back to this problem later.





