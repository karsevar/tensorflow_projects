###Pro Deep learning chapter 4: Natural Language processing: 
##Continuous Bag of Words Implemenation in Tensorflow: 
import numpy as np 
import tensorflow as tf 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import os

emb_dims = 128 
learning_rate = 0.01   

def one_hot(ind, vocab_size):
	rec = np.zeros(vocab_size) 
	rec[ind] = 1
	return rec 

def create_training_data(corpus_raw, WINDOW_SIZE = 2):
	words_list = [] 

	for sent in corpus_raw.split("."): 
		for w in sent.split():
			if w != ".":
				words_list.append(w.split(".")[0]) 

	words_list = set(words_list) 
	word2ind = {} 
	ind2word = {} 

	vocab_size = len(words_list)
	print(len(words_list))  

	for i, w in enumerate(words_list):
		word2ind[w] = i 
		ind2word[i] = w 
		#Build the dictionaries

	print(word2ind)  
	sentences_list = corpus_raw.split(".") 
	sentences = [] 

	for sent in sentences_list: 
		sent_array = sent.split() 
		sent_array = [s.split(".")[0] for s in sent_array]
		sentences.append(sent_array) 
	data_recs = [] 

	for sent in sentences:
		for ind, w in enumerate(sent):
			rec = [] 
			for nb_w in sent[max(ind - WINDOW_SIZE, 0) : min(ind + WINDOW_SIZE,len(sent)) + 1]:
				if nb_w != w:
					rec.append(nb_w) 
				data_recs.append([rec, w]) 
	X_train, y_train = [], []

		  

	for rec in data_recs: 
		input_ = np.zeros(vocab_size)  
		for i in range(WINDOW_SIZE-1):
			input_ += one_hot(word2ind[ rec[0][i] ], vocab_size)
		input_ = input_/len(rec[0]) 
		X_train.append(input_) 
		y_train.append(one_hot(word2ind[ rec[1] ], vocab_size))

	return X_train, y_train, word2ind, ind2word, vocab_size 

corpus_raw = open("data.txt").read()#I was forced to create a text file for the corpus raw example 
#text. It seems that this work around works perfectly. 
corpus_raw = (corpus_raw).lower()

X_train, y_train, word2ind, ind2word, vocab_size = create_training_data(corpus_raw, 2)

emb_dims = 128 
learning_rate = 0.001 

#construction phase tensorflow:
X = tf.placeholder(tf.float32, [None, vocab_size]) 
y = tf.placeholder(tf.float32, [None, vocab_size]) 

W = tf.Variable(tf.random_normal([vocab_size, emb_dims], mean = 0.0, stddev=0.02, dtype=
	tf.float32)) 
b = tf.Variable(tf.random_normal([emb_dims], mean=0.0, stddev=0.02, dtype=tf.float32)) 
W_outer = tf.Variable(tf.random_normal([emb_dims, vocab_size], mean=0.0,stddev=0.02, dtype=tf.float32)) 
b_outer = tf.Variable(tf.random_normal([vocab_size], mean=0.0, stddev=0.02, dtype=tf.float32))

hidden = tf.add(tf.matmul(X, W), b) 
logits = tf.add(tf.matmul(hidden, W_outer), b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

epochs, batch_size = 100,10 
batch = len(X_train)//batch_size
print(batch) 

#Evalutation phase:
init = tf.global_variables_initializer() 

with tf.Session() as sess:
	sess.run(init) 
	for epoch in range(epochs):
		batch_index = 0 
		for batch_num in range(batch):
			X_batch = X_train[batch_index: batch_index + batch_size]
			y_batch = y_train[batch_index: batch_index + batch_size]
			sess.run(optimizer, feed_dict={X: X_batch, y:y_batch})
			print("epoch:", epoch, "loss:", sess.run(cost, feed_dict={X: X_batch, y:y_batch}))
	W_embed_train = sess.run(W)
	#This is kind of a weird implemenation because the aren't the lines for X_batch 
	#and y_batch only sampling the first 10 words within the data set? I believe that the 
	#author will have to use randint() with the indexing command as a means to bring in new 
	#words within the training process. But again I could be wrong. 

W_embedded = TSNE(n_components=2).fit_transform(W_embed_train) 
plt.figure(figsize=(10,10))
for i in range(len(W_embedded)):
	plt.text(W_embedded[i,0],W_embedded[i, 1], ind2word[i]) 

plt.xlim(-150,150)
plt.ylim(-150, 150)
plt.show() 

#The word embeddings learned have been projected to a 2D plane through the TSNE plot. The TSNE
#plot gives a rough idea of the neighborhood of a given word. We can see that the word-embedding vectors 
#learned are reasonable. 

##Skip-gram Implemenation in Tensorflow:
def one_hot(ind, vocab_size):
	rec = np.zeros(vocab_size) 
	rec[ind] = 1
	return rec 

def create_training_data(corpus_raw, WINDOW_SIZE = 2):
	words_list = [] 

	for sent in corpus_raw.split("."): 
		for w in sent.split():
			if w != ".":
				words_list.append(w.split(".")[0]) 

	words_list = set(words_list) 
	word2ind = {} 
	ind2word = {} 

	vocab_size = len(words_list) 

	for i, w in enumerate(words_list):
		word2ind[w] = i 
		ind2word[i] = w 
		#Build the dictionaries

	print(word2ind)  
	sentences_list = corpus_raw.split(".") 
	sentences = [] 

	for sent in sentences_list: 
		sent_array = sent.split() 
		sent_array = [s.split(".")[0] for s in sent_array]
		sentences.append(sent_array) 

	data_recs = [] 

	for sent in sentences:
		for ind, w in enumerate(sent):
			for nb_w in sent[max(ind - WINDOW_SIZE, 0) : min(ind + WINDOW_SIZE,len(sent)) + 1]:
				if nb_w != w:
					data_recs.append([w,nb_w])
	print(data_recs)   
	X_train, y_train = [], []

		  

	for rec in data_recs: 
		X_train.append(one_hot(word2ind[rec[0]], vocab_size))
		y_train.append(one_hot(word2ind[rec[1]], vocab_size))

	return X_train, y_train, word2ind, ind2word, vocab_size 
#This is the same preprocessing function as the continuous bag of words 
#approach. Will need to see the similarities with this method and the continuous bags method 
#as well as the different processing techniques that I should memorize for natural language 
#processing.

corpus_raw = open("data.txt").read()
corpus_raw = (corpus_raw).lower() 

X_train, y_train, word2ind, ind2word, vocab_size = create_training_data(corpus_raw, 2) 

emb_dims = 128 
learning_rate = 0.001 

#construction phase for tensorflow computation map:
tf.reset_default_graph() 

X = tf.placeholder(tf.float32, [None, vocab_size]) 
y = tf.placeholder(tf.float32, [None, vocab_size]) 

W = tf.Variable(tf.random_normal([vocab_size, emb_dims], mean=0.0, stddev=0.02, dtype=tf.float32))
b = tf.Variable(tf.random_normal([emb_dims], mean=0.0, stddev=0.02, dtype=tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims, vocab_size], mean=0.0, stddev=0.02, dtype=tf.float32))
b_outer = tf.Variable(tf.random_normal([vocab_size], mean=0.0, stddev=0.02, dtype=tf.float32))

hidden = tf.add(tf.matmul(X, W), b) 
logits = tf.add(tf.matmul(hidden, W_outer), b_outer) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

epochs, batch_size = 100, 10 
batch = len(X_train)//batch_size 

#Evalutation phase Tensorflow computation map:
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("was here")
	for epoch in range(epochs):
		batch_index = 0 
		for batch_num in range(batch):
			X_batch = X_train[batch_index: batch_index + batch_size] 
			y_batch = y_train[batch_index: batch_index + batch_size] 
			sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})
			print("epoch:", epoch, "loss:", sess.run(cost, feed_dict={X:X_batch, y:y_batch})) 
	W_embed_trained = sess.run(W) 

W_emdedded = TSNE(n_components=2).fit_transform(W_embed_trained) 
plt.figure(figsize=(10,10))
for i in range(len(W_embedded)):
	plt.text(W_embedded[i, 0], W_embedded[i, 1], ind2word[i]) 

plt.xlim(-150, 150) 
plt.ylim(-150, 150)

plt.show()#After running this code a couple more times I found that the embedding algorithms 
#are a little random at best. Meaning that outcomes between sessions might vary. Will need to 
#keep an eye on this and see if I can make this method a little more 
#reliable for scaled projects. 

##Global Co-occurance statistics-based on word vectors experiment 
#using singular vector decomposition.

corpus = ["I like Machine Learning.","I like Tensorflow.","I prefer Python."]

corpus_words_unique = set() 

corpus_processed_docs = []
for doc in corpus:
	corpus_words_ = [] 
	corpus_words = doc.split() 
	print(corpus_words) 
	for x in corpus_words:
		if len(x.split(".")) == 2:
			corpus_words_ += [x.split(".")[0]] + ["."] 
		else:
			corpus_words_ += x.split(".") 
	corpus_processed_docs.append(corpus_words_) 
	corpus_words_unique.update(corpus_words_)

corpus_words_unique = np.array(list(corpus_words_unique)) 

co_occurence_matrix = np.zeros((len(corpus_words_unique), len(corpus_words_unique)))
for corpus_words_ in corpus_processed_docs:
	for i in range(1, len(corpus_words_)):
		index_1 = np.argwhere(corpus_words_unique == corpus_words_[i]) 
		index_2 = np.argwhere(corpus_words_unique == corpus_words_[i-1])

		co_occurence_matrix[index_1, index_2] += 1
		co_occurence_matrix[index_2, index_1] += 1

U, S, V = np.linalg.svd(co_occurence_matrix, full_matrices=False)
#I remember seeing this function and operation before in one of Andrew Ng's lectures 
#will need to look into this. 

print("co-occurence_matrix follows:")
print(co_occurence_matrix) 

for i in range(len(corpus_words_unique)):
	plt.text(U[i, 0], U[i,1], corpus_words_unique[i]) 
plt.xlim((-0.75, 0.75))
plt.ylim((-0.75, 0.75))
plt.show() 








