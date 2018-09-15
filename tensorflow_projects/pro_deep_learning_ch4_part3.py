###Pro deep learning chapter 4 part 3 
##Next Word prediction and sentence completion in Tensorflow using recurrent 
#neural networks:
#Very interesting implemenation will need to review 268 for the project overview.

import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import rnn 
import random 
import collections 

learning_rate = 0.001 
training_iters = 50000
display_step = 500 
n_input = 3 

#number of units in RNN cell 
n_hidden = 512 

#function to read and process the input file:
def read_data(file):
	with open(file) as f: 
		data = f.readlines() 
	data = [x.strip() for x in data] 
	data = [data[i].lower().split() for i in range(len(data))]
	data = np.array(data) 
	data = np.hstack(data) 
	return data 

def build_dataset(train_data):
	count = collections.Counter(train_data).most_common() 
	dictionary = dict() 
	for word, _ in count:
		dictionary[word] = len(dictionary) 
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return dictionary, reverse_dictionary 

def input_one_hot(num):
	x = np.zeros(vocab_size) 
	x[num] = 1 
	return x.tolist() 

#read the input file and build the required dictionaries:
train_file = "alice_passage.txt"
train_data = read_data(train_file) 
print(train_data) 
dictionary, reverse_dictionary = build_dataset(train_data) 
vocab_size = len(dictionary) 

X = tf.placeholder("float", [None, n_input, vocab_size]) 
y = tf.placeholder("float", [None, vocab_size]) 

#RNN output node weights and biases:
weights = {
	"out": tf.Variable(tf.random_normal([n_hidden, vocab_size])) 
}
biases = {
	"out": tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(X, weights, biases):
	X = tf.unstack(X, n_input, 1) 

	#2 layer LSTM Definition 
	rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)]) 

	#generate prediction 
	output, states = rnn.static_rnn(rnn_cell, X, dtype=tf.float32) 

	#there are n_input outputs but we only want the last output 
	return tf.matmul(output[-1], weights["out"]) + biases["out"]

pred = RNN(X, weights, biases) 

#loss and optimizer:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) 
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost) 

#model evaluation 
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer() 

#evaluate computation graph:
with tf.Session() as sess:
	sess.run(init) 
	step = 0 
	offset = random.randint(0, n_input+1)
	end_offset = n_input + 1 
	acc_total = 0 
	loss_total = 0 

	while step < training_iters:
		if offset > (len(train_data)-end_offset):
			offset = random.randint(0, n_input+1) 

		symbols_in_keys = [input_one_hot(dictionary[str(train_data[i])]) for i in 
		range(offset, offset+n_input) ]
		symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vocab_size]) 
		symbols_out_onehot = np.zeros([vocab_size], dtype=float) 
		symbols_out_onehot[dictionary[str(train_data[offset+n_input])]] = 1.0 
		symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1]) 

		_, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], 
			feed_dict = {X: symbols_in_keys, y: symbols_out_onehot})
		loss_total += loss 
		acc_total += acc

		if (step+1) % display_step == 0: 
			print("Iter=" + str(step+1) + ", Average Loss=" + 
				"{:.6f}".format(loss_total/display_step))
			acc_total = 0 
			loss_total = 0 
			symbols_in = [train_data[i] for i in range(offset, offset + n_input)] 
			symbols_out = train_data[offset + n_input] 
			symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())] 
			print("%s - Actual word:[%s] vs predicted word:[%s]" % (symbols_in, symbols_out, symbols_out_pred)) 
		step += 1 
		offset += (n_input+1) 
	print("Training completed") 

	#Feed a 3 word sentence and let the model predict the next 28 words:
	sentence = "she fell slowly"
	words = sentence.split(" ") 
	try:
		symbols_in_key = [input_one_hot(dictionary[ str(train_data[i])]) for i in 
		range(offset, offset+n_input) ]
		for i in range(28):
			keys = np.reshape(np.array(symbols_in_key), [-1, n_inpu, vocab_size]) 
			onehot_pred = sess.run(pred, feed_dict={X: keys})
			onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
			sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index]) 
			symbols_in_keys = symbols_in_keys[1:] 
			symbols_in_keys.append(input_one_hot(onehot_pred_index))
		print("Complete sentence follows")
		print(sentence)
	except:
		print("error while processing the sentence to be completed") 


