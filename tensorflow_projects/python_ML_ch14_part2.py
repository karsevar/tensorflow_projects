##LSTM Cell
import tensorflow as tf 
import numpy as np 
from sklearn.datasets import fetch_mldata 
from sklearn.model_selection import train_test_split 

#Also unknown as the long short term memory cell. If you consider the LSTM
#cell as a black box, it can be used very much like a basic cell, except it will perform 
#much better; training will converge faster and it will detect long-term 
#dependencies in the data. In tensorflow all you need to do is call the 
#tf.contrib.rnn.BasicLSTMCell() function in place of the regular 
#tf.contrib.rnn.BasicRNNCell().

#LSTM cells manage two state vectors, and for performance reasons they are 
#kept separate by default. You can change this default behavior by setting 
#the state_is_tuple arguement to False.  

#To see the full architecture of a LSTM cell look at page 404. 

##Peephole connections: 
#In a basic LSTM cell, the gate controllers can look only at the input X_(t)
#and the previous short-term state h_(t-1). It may be a good idea to give them 
#a bit more context by letting them peek at the long term state as well. 

#Look at page 405 to see the technical implementations of this method. 

#To implement this method in tensorflow, you must use the LSTMCell instead of 
#the BasicLSTMCell() and set use_peepholes to True. 

##GRU cell: 
#The Gated Recurrent Unit cell. The GRU cell is a simplified version of the 
#LSTM cell, and it seems to perform just as well.
#Look at page 406-407 to see the main simplification compared to the 
#basic LSTM cells.

#LSTM technical exercise (most likely the author is using the mnist dataset from earlier): 
tf.reset_default_graph() 

n_steps = 28 
n_inputs = 28 
n_neurons = 150 
n_outputs = 10 
n_layers = 3 

learning_rate = 0.001 

mnist = fetch_mldata("MNIST original") 
X_mnist, y_mnist = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, 
															random_state = 42, shuffle = True)
X_train = X_train.reshape((-1, n_steps, n_inputs))  
X_test = X_test.reshape((-1, n_steps, n_inputs)) 

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) 
y = tf.placeholder(tf.int32, [None]) 

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons) 
				for layer in range(n_layers)] 
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells) 
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype = tf.float32) 
top_layer_h_state = states[-1][1] 
logits = tf.layers.dense(top_layer_h_state, n_outputs, name = "softmax") 
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits) 
loss = tf.reduce_mean(xentropy, name = "loss") 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)  
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1) 
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 

init = tf.global_variables_initializer() 

print(states) 
print(top_layer_h_state) 

n_epochs = 1 
batch_size = 150
n_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))
print(n_batches)  

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
	return X_batch, y_batch

with tf.Session() as sess: 
	init.run() 
	for epoch in range(n_epochs):
		for batch_index in range(n_batches): 
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict = {X: X_batch, y:y_batch}) 
		acc_train = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict = {X: X_test, y:y_test}) 
		print(epoch, "Train accuracy:", acc_train, "Test_accuracy:", acc_test)
#Note to self, LSTM cells are very slow to train. I will need to use a better 
#device to run this code snippet as my current machine is a macbook air. XD
#had to set the number of epochs to one because of the slow convergence rate.

##Natural Language Processing: 
#To see the entire process of vocabulary embedding, look at pages 
#407 through 411. 

##Embeddings english to french translator tutorial: 

#Fetch the data:
from six.moves import urllib 

import errno 
import os 
import zipfile 

WORDS_PATH = "datasets/words"
WORDS_URL = "http://mattmahoney.net/dc/text8.zip"

def mkdir_p(path):
	os.makedirs(path, exis_ok = True) 

def fetch_words_data(words_url=WORDS_URL, words_path = WORDS_PATH):
	os.makedirs(words_path, exist_ok = True) 
	zip_path = os.path.join(words_path, "words.zip") 
	if not os.path.exists(zip_path):
		urllib.request.urlretrieve(words_url, zip_path) 
	with zipfile.ZipFile(zip_path) as f:
		data = f.read(f.namelist()[0]) 
	return data.decode("ascii").split() 

words = fetch_words_data() 

print(words[:5]) 

##Build the dictionary: 
from collections import Counter 

vocabulary_size = 50000

vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)#The command [UNK] 
#was explained in page 408. I believe this effectively tokenizes the phrases within the 
#the inputted word document. 

vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words]) 
print(" ".join(words[:9]), data[:9]) 
print(" ".join([vocabulary[word_index] for word_index in [5241, 3081, 12, 6, 195, 2, 3134, 46, 59]]))
print(words[24], data[24]) 

##Generate batches:
import random 
from collections import deque 

def generate_batch(batch_size, num_skips, skip_window):
	global data_index 
	assert batch_size % num_skips == 0 
	assert num_skips <= 2 * skip_window 
	batch = np.ndarray(shape=(batch_size), dtype = np.int32)#embedding strings are 
	#coded as integers good to know.
	labels = np.ndarray(shape=(batch_size, 1), dtype =np.int32)
	span = 2 * skip_window + 1 
	buffer = deque(maxlen=span) 
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data) 
	for i in range(batch_size // num_skips):
		target = skip_window 
		targets_to_avoid = [ skip_window ] 
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1) 
			targets_to_avoid.append(target) 
			batch[i * num_skips + j] = buffer[skip_window] 
			labels[i * num_skips + j, 0] = buffer[target] 
		buffer.append(data[data_index]) 
		data_index = (data_index + 1) % len(data) 
	return batch, labels 

data_index = 0 
batch, labels = generate_batch(8, 2, 1) 
print(batch, [vocabulary[word] for word in batch]) 
print(labels, [vocabulary[word] for word in labels[:, 0]]) 

##Build the model:
batch_size = 128 
embedding_size = 128 
skip_window = 1 
num_skips = 2 

valid_size = 16 
valid_window = 100 
valid_examples = np.random.choice(valid_window, valid_size, replace=False) 
num_sampled = 64 

learning_rate = 0.01 

tf.reset_default_graph() 

train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1]) 
valid_dataset = tf.constant(valid_examples, dtype = tf.int32) 

vocabulary_size = 50000 
embedding_size = 150 

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0) 
embeddings = tf.Variable(init_embeds) 

train_inputs = tf.placeholder(tf.int32, shape = [None])
embed = tf.nn.embedding_lookup(embeddings, train_inputs) 

#Construct the variables for the NCE loss:
nce_weights = tf.Variable(
	tf.truncated_normal([vocabulary_size, embedding_size],
						stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

#Compute the average NCE loss for the batch.
#tf.nce_loss automatically draws a new sample of the negative labels 
#each time we evaluate the loss.
loss = tf.reduce_mean(
	tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled,
					vocabulary_size))

#construct the adam optimizer:
optimizer = tf.train.AdamOptimizer(learning_rate) 
training_op = optimizer.minimize(loss) 

#compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis = 1, keep_dims = True))
normalized_embeddings = embeddings / norm 
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) 
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) 

init = tf.global_variables_initializer() 

num_steps = 10001

with tf.Session() as sess:
	init.run() 

	average_loss = 0 
	for step in range(num_steps):
		print("\rIteration: {}".format(step), end = "\t") 
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window) 
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = sess.run([training_op, loss], feed_dict=feed_dict)
		average_loss += loss_val 

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000 
			#The average loss is an estimate of the loss over the last 2000 batches.
			print("average loss at step ", step, ": ", average_loss) 
			average_loss = 0 

		#Note that this is expensize (-20% slowdown if computed every 500 steps) 
		if step % 10000 == 0:
			sim = similarity.eval() 
			for i in range(valid_size):
				valid_word = vocabulary[valid_examples[i]]
				top_k = 8 #number of nearest neighbors.
				nearest = (-sim[i, :]).argsort()[1:top_k+1] 
				log_str = "Nearest to %s:" % valid_word
				for k in range(top_k):
					close_word = vocabulary[nearest[k]]
					log_str = "%s %s," % (log_str, close_word) 
				print(log_str) 

	final_embeddings = normalized_embeddings.eval() 

np.save("./my_final_embeddings.npy", final_embeddings) 

##plot the embeddings:
def plot_with_labels(low_dim_embs, labels):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"









