###Fundamentals of deep learning chapter 7:
##Dissecting a neural translation network that uses attention and 
##bucket padding:

#code of the bucket padding technique as well as reversing the word 
#positions within the input sequence (since usually in human languages the 
#later words are more important than the initial words.) 
def get_batch(self, data, bucket_id):
	encoder_size, decoder_size = self.buckets[buckets_id] 
	encoder_inputs, decoder_inputs = [], [] 

	for _ in range(self.batch_size):
		encoder_input, decoder_input = random.choice(data[bucket_id]) 

		#encoder inputs are padded and then reversed.
		encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input)) 
		encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

		#Decoder inputs get an extra "GO" symbol, and are then padded using the 
		#same scheme as the encoder input data.
		decoder_pad_size = decoder_size - len(decoder_input) - 1 
		decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size) 

	#New we create batch.major vectors from the data selected above:
	batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], [] 

	#Batch encoder inputs are just re-indexed encoder_inputs: 
	for length_idx in range(encoder_size):
		batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in range(self.batch_size)],
			dtype=np.int32))

	for length_idx in range(decoder_size):
		batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in range(self.batch_size)], dtype=np.int32))  

		#Create target weights to be 0 for targets that are padded.
		batch_weight = np.ones(self.batch_size, dtype=np.float32) 
		for batch_idx in range(self.batch_size):
			#We set weight to 0 if the corresponding target is a PAD symbol.
			#The corresponding target is decoder_input shifted by 1 forward. 
			if length_idx < decoder_size - 1: 
				target = decoder_inputs[batch_idx][length_idx + 1] 
			if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
				batch_weight[batch_idx] = 0.0 
			batch_weights.append(batch_weight)

	return batch_encoder_inputs, batch_decoder_inputs, batch_weights 

def train():
	"""Train a en>fr translation model using WMT data."""
	#prepare WMT data.
	print("Preparing WMT data in %s" % FLAGS.data_dir) 
	en_train, ft_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocad_size) 

	with tf.Session() as sess:
		#create model.
		print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size)) 
		model = create_model(sess, False) 

		#read data into bucksts and compute their sizes. 
		print("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
		dev_set = read_data(en_dev, fr_dev) 
		train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size) 
		train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))] 

		#training loop.
		step_time, loss = 0.0, 0.0 
		current_step = 0 
		previous_losses = [] 

		while True: 
			#Choose a bucket according to data distribution. We pick a random number in 
			#[0,1] and use the corresponding interval in train_bucket_scale. 

			random_number_01 = np.random.random_sample() 
			bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] 
				> random_number_01]) 
			#Get a batch and make a step.
			start_time = time.time() 
			encoder_inputs, decoder_inputs, target_weights = get_batch(train_set, bucket_id) 
			_, step_loss, _ = step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, 
				False) 
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint 
			loss += step_loss / FLAGS.steps_per_checkpoint 
			current_step += 1 

			#Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				#Print statistics for the previous epoch.
				perplexity = math.exp(float(loss)) if loss < 300 else float("inf") 
				print("global step %d learning rate %.4f step-time %.2f perplexity %.2f" %(model.global_step.eval(),
					model.learning_rate.eval(), step_time, perplexity)) 
				#Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op) 
				previous_losses.append(loss) 
				#Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt") 
				model.saver.save(sess, checkpoint_path, global_step = model.global_step) 
				step_time, loss = 0.0, 0.0 

				#Run evals on development set and print their perplexity:
				for bucket_id in range(len(_buckets)):
					if len(dev_set[bucket_id]) == 0:
						print("eval: empty bucket %d" % (bucket_id))
						continue
					encoder_inputs, decoder_inputs, target_weights = get_batch(dev_set, bucket_id) 
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True) 
					eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
					print(" eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx)) 
				sys.stdout.flush() 

		#Creating a new session that can create the metrics of a newly disclosed sequence and 
		#outputing a newly translated phrase.
def decode():
	with tf.Session() as sess:
	#The following function creates a new tensorflow session and enocodes the sequence 
	#and decodes the sequence to the targeted language. 
	#Create model and load parameters.
	model = create_model(sess, True) 
	batch_size = 1 
	en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.en" % FLAGS.en_vocab_size) 
	fr_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.fr" % FLAGS.fr_vocab_size) 
	en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path) 
	_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path) 

	#I wonder for user input functionality can you use the input() function in place of the 
	#proposed input function in the book page 202. 
	#Decode from standard input. 
	sys.stdout.write("> ") 
	sys.stdout.flush() 
	sentence = sys.stdin.readline() 

	while sentence: 
		#get token ids for the input sentence.
		token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab) 

		#which bucket does it belong to?
		buck_id = len(_buckets) - 1 
		for i, bucket in enumerate(_buckets):
			if bucket[0] >= len(token_ids):
				bucket_id = i 
				break
			else: 
				logging.warning("sentence truncated: %s", sentence) 
			#Get a 1-element batch to feed the sentence to the model
			encoder_inputs, decoder_inputs, target_weights = get_batch({bucket_id: [(token_ids, [])]}, bucket_id) 
			#Instead of only obtaining the log loss properties of this algorithm we want 
			#to hard code this section to out put the translation of the sequence (the unnormalized
			#log probabilities of the output tokens). 
			#Get output logits of the sentence.
			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
							bucket_id, True) 
			#This is a greedy decoder - outputs are just argmaxes of output_logits.
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
			#If there is an EOS symbol in outputs, cut them at that point.
			if data_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(data_utils.EOS_ID)] 
			#Print out French sentence that corresponds to outputs 
			print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
			print("> ", end="") 
			sentence = sys.stdin.readlines() 

#Keep in mind that this extension is only a note taking exercise and has completely no 
#functionality what so ever. As such this function is supposed to be within a class named 
#Seq2Seq within the seq2seq_model.py module.
def step(self, session, encoder_inputs, decoder_inputs, target_weights,
	bucket_id, forward_only):
	#check if the sizes match. 
	encoder_size, decoder_size = self.buckets[bucket_id] 
	if len(encoder_inputs) != encoder_size:
		raise ValueError("Encoder length must be equal to the one in bucket," " %d != %d." %(len(encoder_inputs), 
			encoder_size)) 

	if len(decoder_inputs) != decoder_size:
		raise ValueError("Decoder length must be equal to the one in bucket," " %d != %d." %(len(decoder_inputs), 
			decoder_size)) 

	if len(target_weights) != decoder_size:
		raise ValueError("Weights length must be equal to the on in bucket," " %d != %d." %(len(target_weights), 
			decoder_size))

	#Input feet: encoder inputs, decoder inputs, target_weights, as provided.
	input_feed = {} 
	for l in xrange(encoder_size):
		input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
	for l in range(decoder_size):
		input_feed[self.decoder_inputs[l].name] = decoder_inputs[l] 
		input_feed[self.target_weights[l].name] = target_weights[l] 

	#Since our targets are decoder inputs shifted by one. we need one more.
	last_target = self.decoder_inputs[decoder_size].name 
	input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32) 

	#Output feed: depends on whether we do a backward step or not. 
	if not forward_only:
		output_feed = [self.updates[bucket_id], 
		self.gradient_norms[bucket_id],
		self.losses[bucket_id]]
	else:
		output_feed = [self.losses[bucket_id]]
		for l in range(decoder_size):
			output_feed.append(self.outputs[bucket_id][l]) 

	outputs = session.run(output_feed, input_feed)
	if not forward_only:
		return outputs[1], outputs[2], None
	else:
		return None, outputs[0], outputs[1:] 
		#No gradient norm, loss, outputs.

#The following function call can be found in the translate.py module. Again 
#this session is more about notes and scholarship than functionality. 
def create_model(session, forward_only):
	"""create translation model and initialize or load 
	parameters in session"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	model = seq2seq_model.Seq2SeqModel(
		FLAGS.en_vocab_size, 
		FLAGS.fr_vocab_size, 
		_buckets,
		FLAGS.size,
		FLAGS.num_layers, 
		FLAGS.max_gradient_norm,
		FLAGS.batch_size, 
		FLAGS.learning_rate, 
		FLAGS.learning_rate_decay_factor, 
		forward_only=forward_only,
		dtype=dtype) 

	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir) 
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path) 
		model.saver.restore(session, ckpt.model_checkpoint_path) 
	else:
		print("Created model with fresh parameters") 
		session.run(tf.global_variables_initializer())
	return model 

#the following is the Seq2SeqModel() class object which can be found in 
#the seq2seq_model.py module. 

class Seq2SeqModel(object):
	def __init__(self, 
				source_vocab_size, 
				target_vocab_size, 
				buckets, 
				size, 
				num_layers, 
				max_graident_norm,
				batch_size, 
				learning_rate,
				learning_rate_decay_factor, 
				use_lstm=False, 
				num_samples=512, 
				forward_only=False,
				dtype=tf.float32):
		self.source_vocab_size = source_vocab_size
		self.target_vocab_size = target_vocab_size 
		self.buckets = buckets 
		self.batch_size = batch_size 
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor) 
		self.global_step = tf.Variable(0, trainable=False) 

		#If we use sampled softmax, we need an output projection.
		output_projection = None 
		softmax_loss_function = None 
		#Sampled softmax only makes sence if we sample less than vocabulary size.
		if num_samples > 0 and num_samples < self.target_vocab_size:
			w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype) 
			w = tf.transpose(w_t) 
			b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype) 
			output_projection = (w, b) 

			def sampled_loss(inputs, labels):
				labels_w_t = tf.cast(w_t, tf.float32) 
				#We need to compute the sampled softmax loss using 32bit foats to 
				#not incure errors. 
				local_w_t = tf.cast(w_t, tf.float32) 
				local_b = tf.cast(b, tf.float32) 
				local_inputs = tf.cast(inputs, tf.float32) 
				return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
					num_samples, self.target_vocab_size), dtype)

			softmax_loss_function = sampled_loss 

		#Create the internal multi_layer cell for our RNN.
		single_cell = tf.nn.rnn_cell.GRUCell(size) 
		if use_lstm: 
			single_cell = tf.nn.rnn_cell.GRU(size)
		cell = single_cell 
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers) 

		#The seq2seq function: we use embedding for the input and attention 
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return seq2seq.embedding_attention_seq2seq(encoder_inputs, 
													decoder_inputs,
													cell, 
													num_encoder_symbols=source_vocab_size,
													num_decoder_symbols=target_vocab_size,
													embedding_size=size,
													output_size=size,
													output_projection=output_projection,
													feed_previous=do_decode,
													dtype=dtype) 
		#feeds for inputs.
		self.encoder_inputs = [] 
		self.decoder_inputs = [] 
		self.target_weights = [] 
		for i in range(buckets[-1][0]): 
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
				name="encoder{0}".format(i)))

		for i in range(buckets[-1][1] + 1):
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
				name="decoder{0}".format(i)))

			self.target_weights.append(tf.placeholder(dtype, shape=[None],
				name="weight{0}".format(i)))

		targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

		#Training outputs and losses:
		if forward_only:
			self.outputs, self.losses = seq2seq.model_with_buckets(self.encoder_inputs,
				self.decoder_inputs, targets, self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
				softmax_loss_function=softmax_loss_function)
			#If we use output projection, we need to project outputs for decodeing.
			if output_projection is not None:
				for b in range(len(buckets)):
					self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]]

		else: 
			self.outputs, self.losses = seq2seq.model_with_buckets(self.encoder_inputs,
				self.decoder_inputs, targets, self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, False),
				softmax_loss_function=softmax_loss_function) 

		#Gradients and SGD update operation for training the model.
		params = tf.trainable_variables() 
		if not forward_only:
			self.gradient_norms = [] 
			self.updates = [] 
			opt = tf.train.GradientDescentOptimizer(self.learning_rate) 
			for b in range(len(buckets)):
				gradients = tf.gradients(self.losses[b], params) 
				clipped_gradients, norm = tf.clip_by_global_norm(gradients, 
					max_gradient_norm) 
				self.gradient_norms.append(norm) 
				self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
		self.saver = tf.train.Saver(tf.all_variables())


#This following step will hard code the attention layer as well as create the 
#the key characteristic that makes this model an autoregressive decoder. 
#This function can be found in the seq2seq.py module.
def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
	if output_size is None:
		output_size = cell.output_size 
	if output_projection is not None:
		proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
		proj_biases.get_shape().assert_is_compatible_with([num_symbols]) 

	with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope;
		embedding = variable_scope.get_variable("embedding", 
			[um_symbols, embedding_size]) 
		loop_function = _extrat_argmax_and_embed(embedding, 
			output_projection, update_embedding_for_previous) if feed_previous else None 
		emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
		return attention_decoder( 
			emb_inp, 
			initial_state,
			attention_states,
			cell,
			output_size=output_size,
			loop_function=loop_function,
			initial_state_attention=initial_state_attention) 


def embedding_attention_seq2seq(encoder_inputs, 
								decoder_inputs,
								cell, 
								num_encoder_symbols,
								num_decoder_symbols,
								embedding_size, 
								output_projection=None,
								feed_previous=False,
								dtype=None,
								scope=None,
								initial_state_attention=False):
	with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
		dtype = scope.dtype
		#Encoder.
		encoder_cell = rnn_cell.EmbeddingWrapper(cell,
			embedding_classes=num_encoder_symbols,
			embedding_size=embedding_size) 
		encoder_outputs, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype) 

		#first calculate a concatenation of encoder outputs to put attention on. 
		top_states = [array_ops.reshape(e, [-1, 1. cell.output_size]) for e in encoder_outputs]

		#attention_states = array_ops.concat(top_states,1)
		attention_states = array_ops.concat(1, top_states)

		#Decoder.
		output_size = None 
		if output_projection is None:
			cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols) 
			output_size = num_decoder_symbols

		if isinstance(feed_previous, bool):
			return embedding_attention_decoder(
				decoder_inputs,
		        encoder_stat,
		        attention_states,
		        cell,
		        num_decoder_symbols,
		        embedding_size,
		        output_size=output_size,
		        output_projection=output_projection,
		        feed_previous=feed_previous,
		        initial_state_attention=initial_state_attention) 

















