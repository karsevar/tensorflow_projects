###Pro Deep learning chapter 3: convolution Neural Networks
##Convolutional Neural Network for Solving Real-World Problems 
#Interesting the author uses a kaggle problem posted by Intel 
#that had the pariticipants create a classification CNN that can 
#classify different forms of cervix cancer types. The following is 
#a brief solution to this problem.

import tensorflow as tf
import matplotlib.pyplot as plt 
from PIL import ImageFilter, ImageStat, Image, ImageDraw 
from multiprocessing import Pool, cpu_count 
from sklearn.preprocessing import LabelEncoder 
import pandas as pd 
import numpy as np 
import glob
import cv2 
import time 
from keras.utils import np_utils 
import os 

#Read the input images and then resize the image to 64 by 64 
#by 3 size (will need to remember this command. This might fix the 
#problem that I've been having with the cifar dataset). 
def get_im_cv2(path):
	img = cv2.imread(path) 
	resized = cv2.resize(img, (64,64), cv2.INTER_LINEAR) 
	return resized

#Each of the folders corresponds to a different class 
#load the images into array and then define their output 
#classes based on the folder number. 

#Training set 
def load_train(): 
	X_train = []
	X_train_id = [] 
	y_train = [] 
	start_time = time.time() 

	print("read train images") 
	folders = ["Type_1", "Type_2", "Type_3"] 
	for fld in folders: 
		index = folders.index(fld)
		print("Load folder {} (Index: {})".format(fld, index)) 
		path = os.path.join(".", "Downloads", "Intel","train",fld,"*.jpg") 
		files = glob.glob(path) 

		for fl in files: 
			flbase = os.path.basename(fl) 
			img = get_im_cv2(fl) 
			X_train.append(img) 
			X_train_id.append(flbase) 
			y_train.append(index) 

	for fld in folders:
		index = folders.index(fld) 
		print("load folder {} (Index: {})".format(fld, index))
		path = os.path.join(".", "Downloads", "Intel","Additional",fld,"*.jpg") 
		files = glob.glob(path) 

		for fl in files:
			flbase = os.path.basename(fl) 
			img = get_im_cv2(fl) 
			X_train.append(img) 
			X_train_id.append(flbase) 
			y_train.append(index) 

	print("Read train data time: {} seconds".format(round(time.time() - start_time, 2)))
	return X_train, y_train, X_train_id 

#Testing set:
def load_test():
	path = os.path.join(".","Downloads","INtel","test","*jpg") 
	files = sorted(glob.glob(path)) 

	X_test = [] 
	X_test_id = [] 
	for fl in files:
		flbase = os.path.basename(fl) 
		img = get_im_cv2(fl) 
		X_test.append(img) 
		X_test_id.append(flbase) 
	path = os.path.join(",", "Downloads","Intel","test_stg2","*.jpg") 
	files = sorted(glob.glob(path)) 
	for fl in files:
		flbase = os.path.basename(fl) 
		img = get_im_cv2(fl) 
		X_test.append(img) 
		X_test_id.append(flbase) 

	return X_test, X_test_id 

#normalize the image data to have values between 0 and 1 
#through dividing each array by 255 and convert the class label 
#into a one hot array.

def read_and_normalize_train_data():
	train_data, train_target, train_id = load_train() 

	print("convert to numpy...") 
	train_data = np.array(train_data, dtype=np.uint8)
	train_target = np.array(train_target, dtype=np.uint8)

	print("Reshape...")
	train_data = train_data.transpose((0,2,3,1))
	train_data = train_data.transpose((0,1,3,2))

	print("Convert to float...") 
	train_data = train_data.astype("float32") 
	train_data = train_data / 255
	train_target = np_utils.to_categorical(train_target, 3) 

	print("Train shape:", train_data.shape) 
	print(train_data.shape[0], "train samples") 
	return train_data, train_target, train_id 

def read_and_normalize_test_data():
	start_time = time.time() 
	test_data, test_id = load_test() 

	test_data = np.array(test_data, dtype=np.uint8) 
	test_data = test_data.transpose((0,2,3,1))
	test_data = test_data.transpose((0,1,3,2))

	test_data = test_data.astype("float32") 
	test_data = test_data / 255

	print("Test shape:", test_data.shape) 
	print(test_data.shape[0], "test samples") 
	print("read and process test data time: {} seconds".format(round(time.time() - 
		start_time, 2)))
	return test_data, test_id 

#Read and normalize the train data 
#train_data, train_target, train_id = read_and_normalize_train_data() 

#shuffle the input training data to aid stochastic gradent descent 
#Might have to skip this part due to me being unable to install the shuffle 
#module. Will most likely shuffle the training data with my fetch_batch 
#function.

channel_in = 3
channel_out = 64 
channel_out1 = 128 

#convolution layer 
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x,W,strides=[1, strides, strides, 1], padding="SAME") 
	x = tf.nn.bias_add(x, b) 
	return tf.nn.relu(x)

#max pool layer:
def maxpool2d(x, stride=2):
	return tf.nn.max_pool(x,ksize=[1,stride,stride,1], strides=[1,stride,stride,1], padding="SAME")

#The Neural Network architecture:
def conv_net(x, weights, biases, dropout):
	 
	#convolutional layer 1
	conv1 = conv2d(x, weights["wc1"],biases["bc1"]) 
	conv1 = maxpool2d(conv1, 2)

	#Convolutional layer 2
	conv2a = conv2d(conv1,weights["wc2"],biases["bc2"]) 
	conv2a = maxpool2d(conv2a,stride=2)

	conv2 = conv2d(conv2a, weights["wc3"], biases["bc3"]) 
	conv2 = maxpool2d(conv2, stride=2)  

	#fully connected layer
	fc1 = tf.reshape(conv2,[-1,weights["wd1"].get_shape().as_list()[0]]) 
	fc1 = tf.add(tf.matmul(fc1,weights["wd1"]),biases["bd1"]) 
	fc1 = tf.nn.relu(fc1)

	#apply dropout:
	fc1 = tf.nn.dropout(fc1,dropout) 

	#output class prediction:
	out = tf.add(tf.matmul(fc1,weights["out"]),biases["out"]) 
	return out	 

#Define several parameters for the network and learning 
learning_rate = 0.01 
epochs = 200 
batch_size = 128 
num_batches = X_train.shape[0]/128
input_height = 64
input_width = 64 
n_classes = 3
dropout = 0.5 
display_step = 1 
filter_height = 3
filter_width = 3
depth_in = 3
depth_out1 = 64
depth_out2 = 128 
depth_out3 = 256 

#Input-output definition 
x = tf.placeholder(tf.float32, [None, input_height_width, depth_in]) 
y = tf.placeholder(tf.float32, [None, n_classes]) 
keep_prob = tf.placeholder(tf.float32) 

weights = {
	"wc1": tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),
	"wc2": tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),
	"wc3": tf.Variable(tf.random_normal([filter_height, filter_width, depth_out2, depth_out3])),
	"wd1": tf.Variable(tf.random_normal([int((input_height/8)*(input_height/8)*256),512])),
	"wd2": tf.Variable(tf.random_normal([512,512])),
	"out": tf.Variable(tf.random_normal([512, n_classes]))
} 

biases = {
	"bc1": tf.Variable(tf.random_normal([64])),
	"bc2": tf.Variable(tf.random_normal([128])),
	"bc3": tf.Variable(tf.random_normal([256])),
	"bd1": tf.Variable(tf.random_normal([512])),
	"bd2": tf.Variable(tf.random_normal([512])),
	"out": tf.Variable(tf.random_normal([n_classes]))
}

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(train_data), size = batch_size) 
	X_batch = train_data[indices] 
	y_batch = train_target[indices] 
	return X_batch, y_batch 


##defining the tensorflow ops for different activaties:
pred = conv_net(x, weights, biases, keep_prob) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

#evaluate model: 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initializing all variables 
init = tf.global_variables_initializer() 

#Evaluation phase (computation graph)
with tf.Session() as sess:
	sess.run(init) 
	for epoch in range(epochs):
		for batch_index in range(num_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(optimizer, feed_dict={x:X_batch, y: y_batch, keep_prob: 1.})
			loss,acc = sess.run([cost,accuracy], feed_dict={x:X_batch, y:y_batch,keep_prob: 1.}) 
			if epochs % display_step == 0: 
				print("Epoch:", "%04d" % (epoch+1),
					"cost=", "{:.9f}".format(loss),
					"Training_accuracy","{:.5f}".format(acc))
	print("optimization completed")

	y1 = sess.run(pred, feed_dict={x:X_test, y:y_test, keep_prob: 1}) 
	test_classes = np.argmax(y1, 1) 
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:X_test[:256], y:y_test[:256], keep_prob: 1}))
	f, a = plt.subplots(1, 10, figsize=(10, 2))





