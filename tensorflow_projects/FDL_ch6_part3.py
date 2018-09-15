###Fundamentals of deep learning chapter 6 part 3:
###When Context is More Informative than the Input Vector:

##The Word2Vec Framework:
#The Word2Vec frameworks (pioneered by mikolov):
#The continous words in bag method and the skip gram method.
#these models are already explained in great detail in the 
#book pro deep learning.

import tensorflow as tf 
import input_word_data as data 
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

#This following code line creates the encoder layer of a continous 
#bag of words model or an n-gram model:
	#tf.nn.embedding_lookup(params, ids, partition_strategy="mod",
		#name=None, validate_indices=True) 

#Where params is the embedding matrix, and ids is the tensor of indices 
#we want to look up. 

#As for the decoder, the best way to create one is to use the noise-contrastive 
#estimation (NCE) method. To see the full process look at page 145 
#To use this method through tensorflow use the function:
	#tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled,
		#num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False,
		#partition_strategy="mod", name="nce_loss") 

#The weights should have the same dimensions as the embedding matrix, and 
#the biases should be a tensor with size equal to the vocabulary. The inputs 
#are the results from the embedding lookup, num_sampled is the number of negative samples 
#we use to compute the NCE, and num_classes is the vocabulary size. 

##Implementing the skip gram architecture in Tensorflow:
batch_size = 32
embedding_size = 128 
skip_window = 5
num_skips = 4 
batches_per_epoch = data.data_size*num_skips/batch_size 
training_epochs = 5
neg_size = 64
display_step = 2000 
val_step = 10000 
learning_rate = 0.1 
val_size = 20 
val_dist_span = 500 
val_examples = np.random.choice(val_dist_span, val_size, replace=False) 
top_match = 8 
plot_num = 500 

#The embedding layer:
def embedding_layer(x, embedding_shape):
	with tf.variable_scope("embedding"):
		embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0) 
		embedding_matrix = tf.get_variable("E", initializer=embedding_init)
		return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix

#To compute the nce cost of each training example:
def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y):
	with tf.variable_scope("nce"):
		nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0/(weight_shape[1])**0.5)
		nce_bias_init = tf.zeros(bias_shape)  
		nce_W = tf.get_variable("W", initializer=nce_weight_init) 
		nce_b = tf.get_variable("b", initializer=nce_bias_init) 
		total_loss = tf.nn.nce_loss(nce_W, nce_b, y, embedding_lookup, neg_size,
			data.vocabulary_size) 
		return tf.reduce_mean(total_loss) 

#The training function utilizing stochastic gradient descent:
def training(cost, global_step):
	with tf.variable_scope("training"):
		summary_op = tf.summary.scalar("cost", cost) 
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		training_op = optimizer.minimize(cost, global_step=global_step)
		return training_op, summary_op 

#The following code computes the cosine distance of particular words 
#from other words within the dictionary. 
def validation(embedding_matrix, x_val):
	norm = tf.reduce_sum(embedding_matrix**2, 1, keep_dims=True)**0.5
	normalized = embedding_matrix/norm
	val_embeddings = tf.nn.embedding_lookup(normalized, x_val) 
	cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b=True) 
	return normalized, cosine_similarity 

with tf.variable_scope("skipgram_model"):
	x = tf.placeholder(tf.int32, shape=[batch_size])
	y = tf.placeholder(tf.int32, [batch_size, 1]) 
	val = tf.constant(val_examples, dtype=tf.int32) 
	global_step = tf.Variable(0, name="global_step", trainable=False) 

	e_lookup, e_matrix = embedding_layer(x, [data.vocabulary_size, embedding_size]) 

	cost = noise_contrastive_loss(e_lookup, [data.vocabulary_size, embedding_size], [data.vocabulary_size], y) 

	train_op = training(cost, global_step) 

	val_op = validation(e_matrix, val) 

	sess = tf.Session() 

	init = tf.global_variables_initializer() 

	sess.run(init) 

	step = 0 
	avg_cost = 0 

	for epoch in range(training_epochs):
		for minibatch in range(int(batches_per_epoch)):

			step += 1 

			minibatch_x, minibatch_y = data.generate_batch(batch_size, num_skips, skip_window) 
			feed_dict = {x: minibatch_x, y: minibatch_y}

			_, new_cost = sess.run([train_op, cost], feed_dict=feed_dict) 
			avg_cost += new_cost/display_step 

			if step % display_step ==0:
				print("Elapsed:", str(step), "batches. Cost=", "{:.9f}".format(avg_cost))
				avg_cost = 0 

			if step % val_step == 0:
				_, similarity = sess.run(val_op) 
				for i in range(val_size):
					val_word = data.reverse_dictionary[val_examples[i]]
					neighbors = (-similarity[i, :]).argsort()[1:top_match+1] 
					print_str = "Nearest neighbor of %s:" % val_word
					for k in range(top_match):
						print_str += " %s," % data.reverse_dictionary[neighbors[k]]
					print(print_str[:-1]) 

	final_embeddings, _=sess.run(val_op) 

tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
plot_embeddings = np.asfarray(final_embeddings[:plot_num,:], dtype="float") 
low_dim_embs = tsne.fit_transform(plot_embeddings) 
labels = [reverse_dictionary[i] for i in range(plot_only)]
data.plot_with_labels(low_dim_embs, labels) 









