###Pro deep learning with Tensorflow Chapter 4 part 2 
##Word Analogy with Word Vectors

##GloVe word embedding model experiment:

import numpy as np 
import scipy 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

#loading glove vector 
EMBEDDING_FILE = "glove.6B.300d.txt"
embedding_index = {}
f = open(EMBEDDING_FILE, "r+", encoding="utf-8") 
count = 0 
for line in f:
	if count == 0:
		count = 1
		continue 
	values = line.split()
	word = values[0] 
	coefs = np.asarray(values[1:], dtype="float32") 
	embedding_index[word] = coefs 
f.close() 

print("Found %d word vectors of glove." %len(embedding_index))

king_wordvec = embedding_index["king"]
queen_wordvec = embedding_index["queen"] 
man_wordvec = embedding_index["man"] 
woman_wordvec = embedding_index["woman"] 

pseudo_king = queen_wordvec - woman_wordvec + man_wordvec 
cosine_simi = np.dot(pseudo_king/np.linalg.norm(pseudo_king), 
	king_wordvec/np.linalg.norm(king_wordvec))
print("cosine similarity", cosine_simi) 

tsne = TSNE(n_components=2) 
words_array = [] 
word_list = ["king","queen","man","woman"] 
for w in word_list:
	words_array.append(embedding_index[w])
index1 = list(embedding_index.keys())[0:100]
print(index1)  
for i in range(100):
	words_array.append(embedding_index[index1[i]]) 
words_array = np.array(words_array) 
print(words_array) 
words_tsne = tsne.fit_transform(words_array) 

ax = plt.subplot(111)
for i in range(4):
	plt.text(words_tsne[i, 0], words_tsne[i, 1], word_list[i]) 
plt.xlim((50, 125))
plt.ylim((0,80))
plt.show() 
#I can't seem to get the results to show up on the graph function above. 
#I will have to look into what I did wrong with this these lines later on through 
#my studies as well as the best embedding model style I should use for my start up 
#idea (automated resumes).

##Mnist Digit identification in tensorflow using recurrent neural networks: 
import tensorflow as tf 
from tensorflow.contrib import rnn 
import numpy as np 

from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 

def read_infile():
	n_classes = 10 
	mnist = fetch_mldata("MNIST original") 
	X_mnist, y_mnist = mnist["data"], mnist["target"]
	y_mnist = np.eye(n_classes)[y_mnist.astype(np.int32)]   
	X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2, shuffle=True, random_state = 42)    
	return X_train, y_train, X_test, y_test

n_classes = 10 
X_train, y_train, X_test, y_test = read_infile()
#Again the tutorial mnist dataset implementation doesn't seem to want to work yet again.
#Will need to use this same function and module from hands on machine learning.  

learning_rate = 0.001 
training_iters = 10000
batch_size = 128 
display_step = 50 
num_train = X_train 
num_batches = int(np.ceil(int(X_train.shape[0]) / batch_size) + 1)
epochs = 4

#RNN LSTM Network parameters:
n_input = 28 
n_steps = 28 #time steps 
n_hidden = 128 
n_classes = 10 

#Important for mini-batch gradient descent:
def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch 

#construction of the lstm neuron architecture:
def RNN(X, weights, biases):
	X = tf.unstack(X, n_steps, 1)

	#Define the lstm cell:
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) 

	#Get lstm cell output:
	outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32) 

	#Linear activation using rnn inner loop last output:
	return tf.matmul(outputs[-1], weights["out"]) + biases["out"] 

#tf graph input: 
X = tf.placeholder("float", [None, n_steps, n_input]) 
y = tf.placeholder("float", [None, n_classes]) 

#define weights:
weights = {
	"out": tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	"out": tf.Variable(tf.random_normal([n_classes]))
}

pred = RNN(X, weights, biases) 

#define loss and optimizer:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

#Evaluation statistics:
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#evalutation phase of the computation graph:
init = tf.global_variables_initializer() 

with tf.Session() as sess:
	sess.run(init)
	i = 0 

	while i < epochs:
		for step in range(num_batches):
			X_batch, y_batch = fetch_batch(i, step, batch_size)
			X_batch = X_batch.reshape((batch_size, n_steps, n_input))
			sess.run(optimizer, feed_dict={X: X_batch, y: y_batch}) 
			if (step + 1) % display_step ==0:
				#calculate batch accuracy:
				acc = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
				#calculate batch loss:
				loss = sess.run(cost, feed_dict={X: X_batch, y: y_batch}) 
				print("Epoch:" + str(i+1) + ",step:" + str(step+1) +", Minibatch loss" + 
					"\n{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc))
		i += 1 
	print("optimization finished") 
	test_len = 500 
	test_data = X_test[:test_len].reshape((-1, n_steps, n_input))
	test_label = y_test[:test_len]
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, y: test_label}))
#this LSTM implementation actually worked. I'm really surprised.

##Next Word prediction and sentence completion in Tensorflow using recurrent 
#neural networks:
#Very interesting implemenation will need to review 268 for the project overview.
import random 
import collections 

learning_rate = 0.001 
training_iters = 50000
display_step = 500 
n_inputs = 3 

#number of units in RNN cell 
n_hidden = 512 

#function to read and process the input file:
def read_data(file):
	with open(file) as f: 
		data = f.readlines() 
	data = [x.strip() for x in data] 
	data = [data[i].lower().split() for i in range(len(data))]
	data = np.array(data) 
	data = np.reshape(data, [-1,]) 
	return data 

def build_dataset(train_data):
	count = collections.Counter(list(train_data)).most_common() 
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
	outputs, states = rnn.static_rnn(rnn_cell, X, dtype=tf.float32) 

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







