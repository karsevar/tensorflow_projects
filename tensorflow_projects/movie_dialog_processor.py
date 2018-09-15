###This module will be used to process the cornell movie 
##dialog corpus. The following code is from github repository 
##github.com/monkut/cornell-movie-corpus-processor.

import os 
import random 
from ast import literal_eval 

DELIM = " +++$+++ "

class CornellMovieCorpusProcessor: 

	def __init__(self, lines="movie_lines.txt", conversations="movie_conversations.txt"):
		self.movie_lines_filepath = lines 
		self.movie_conversations = conversations 

	def get_id2line(self):
		"""
		1.read from the movie_lines.txt"
		2. create a dictionary with (key = line_id, value = txt) 
		:return: (dict) {line-id: text, ...} 
		"""
		id2line = {} 
		id_index = 0 
		text_index = 4
		with open(self.movie_lines_filepath, "r", encoding="ascii", errors="ignore") as f: 
			for line in f:
				items = line.split(DELIM) 
				if len(items) == 5:
					line_id = items[id_index] 
					dialog_text = items[text_index] 
					id2line[line_id] = dialog_text 
				print(items)  
		return id2line

	def get_conversations_me(self):
		conv_lines = open("moviequotes_scripts.txt", encoding="ascii", errors="ignore").read().split("\n")
		convs = [] 
		for line in conv_lines[:-1]:
			_line = line.split("+++$+++")[-1][1:-1].replace("'", "").replace(" ", "") 
			convs.append(_line.split(","))
			p
		return(convs)  


	def get_conversations(self):
		"""
		1. Read from 'movie_conversations.txt'
		2. create a list of [list of line_ids] 
		:return: [list of line_ids] 
		"""
		conversation_ids_index = -1 
		conversations = [] 
		with open(self.movie_conversations, 'r', encoding="ascii") as f: 
			for line in f: 
				items = line.split(DELIM)  
				conversation_ids_field = items[conversation_ids_index] 
				conversation_ids = literal_eval(conversation_ids_field)
				conversations.append(conversation_ids)
				print(conversations)  
		return conversations 

	def get_question_answer_set(self, id2line, conversations):
		"""
		want to collect questions and answers?
		(The author really isn't sure if this method will work to train 
		a machine to answer questions)

		:param conversations: (list) collections line ids consisting of a single conversation
		:param id2line: (dict) mapping of line_ids to actual line text 
		:return: (list) questions, (list) answers 
		"""
		questions = []
		answers = [] 

		#this uses a simple method in an attempt to gather questions/answers
		for conversation in conversations:
			if len(conversation) % 2 != 0:
				conversation = conversation[:-1] 

			for idx, line_id in enumerate(conversation):
				if idx % 2 == 0:
					questions.append(id2line[line_id]) 
				else:
					answers.append(id2line[line_id]) 
		return questions, answers 

	def prepare_seq2seq_files(self, questions, answers, output_directory, test_set_size=30000):
		"""
		preparing training test data for 
		the chat bot. Not sure if I really need this command
		"""

		#open files
		train_enc_filepath = os.path.join(output_directory, "train.enc") 
		train_dec_filepath = os.path.join(output_directory, "train.dec") 
		test_enc_filepath = os.path.join(output_directory, "test.enc") 
		test_dec_filepath = os.path.joint(output_directory, "test.dec") 

		train_enc = open(train_enc_filepath, "w", encoding="utf8") 
		train_dec = open(train_dec_filepath, "w", encoding="utf8") 
		test_enc = open(test_enc_filepath, "w", encoding="utf8") 
		test_dec = open(test_dec_filepath, "w", encoding="utf8") 

		#Choose test_set_size number of items to put into testset 
		test_ids = random.sample(range(len(questions)), test_set_size) 

		for i in range(len(questions)):
			if i in test_ids:
				test_enc.write(questions[i] + "\n") 
				test_dec.write(answers[i] + "\n") 
			else: 
				train_enc.write(questions[i] + "\n") 
				train_dec.write(answers[i] + "\n") 

		#close files
		train_enc.close() 
		train_dec.close() 
		test_enc.close() 
		test_dec.close() 
		return train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath   

data_dir = "moviequotes_scripts.txt"
processor = CornellMovieCorpusProcessor(lines=data_dir, conversations=data_dir) 
processor.get_id2line()
processor.get_conversations() 
  
 
