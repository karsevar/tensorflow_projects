###this is primarily a reiterations of the last module except 
##the other one didn't really work for most of the dataset. 
##The following processor is obtained from the suriyadeepan 
##github repository. 

import random 



def get_id2line():
	lines = open("moviequotes_scripts.txt", encoding="ascii",errors="ignore").read().split("\n") 
	id2line = {} 
	for line in lines:
		_line = line.split(" +++$+++ ") 
		if len(_line) == 5:
			id2line[_line[0]] = _line[4] 
	return id2line 

def get_conversations():
	conv_lines = open("moviequotes_scripts.txt", encoding="ascii", errors="ignore").read().split("\n")
	convs = [ ] 
	for line in conv_lines[:-1]:
		_line = line.split(" +++$+++ ")[-1][1:-1].replace("'","") 
		convs.append(_line.split(','))
	print(convs) 
	return convs

def extract_conversations(convs, id2line, path=''):
	idx = 0 
	for conv in convs:
		f_conv = open(path + str(idx)+".txt", "w")
		for line_id in conv:
			f_conv.write(id2line[line_id]) 
			f_conv.write("\n") 
		f_conv.close() 
		idx += 1 

def gather_dataset(convs, id2line):
	questions = []; answers = [] 

	for conv in convs:
		if len(conv) % 2 != 0:
			conv = conv[:-1] 
		for i in range(len(conv)):
			if i%2 == 0:
				questions.append(id2line[conv[i]]) 
			else:
				answers.append(id2line[conv[i]]) 
	return questions, answers 

id2line = get_id2line()
print(id2line)  
convs = get_conversations()
questions, answers = gather_dataset(convs, id2line)






