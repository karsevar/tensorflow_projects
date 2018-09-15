###Pro Deep learning chapter 3: Convolutional Neural Networks 

##Illustrate 2D convolution of images through an example:
#Conceptualizing LSI system filters in a theoretical light.
import scipy.signal 
import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import tensorflow as tf 

channels = 3

image = mpimg.imread("newfoundland1.jpeg")[:,:,:channels]
#plt.imshow(image) 
#plt.axis("off") 
#plt.show()#Very good this function is working perfectly now. Weird that 
#imshow is working with the pyplot module but not pyplot.plot() will need to 
#experiment a little. 

#print(image) 

image_example = np.array([[1,2,3,4,5,6,7],
	[8,9,10,11,12,13,14],
	[15,16,17,18,19,20,21],
	[22,23,24,25,26,27,28],
	[29,30,31,32,33,34,35],
	[36,37,38,39,40,41,42],
	[43,44,45,46,47,48,49]])   

#defining an image-processig kernel:
filter_kernel = np.array([[-1, 1, -1],
	[-2, 3, 1],
	[2, -4, 0]]) 

#convolve the image with the filter kernel through scipy 2D convolution 
#to produce an output image of same dimension as that of the input.

I = scipy.signal.convolve2d(image_example, filter_kernel, mode="same", boundary="fill", fillvalue=0) 
print(I) 

row, col = 7,7

#Rotate the filter kernel twice by 90 degrees to get 180 rotation.

filter_kernel_flipped = np.rot90(filter_kernel, 2) 
#Pad the boundaries of the image with zeroes and fill the rest from the original image 
image1 = np.zeros((9,9)) 
for i in range(row):
	for j in range(col):
		image1[i+1, j+1] = image_example[i,j] 
print(image1) #this for loop superimposes the 7 by 7 image onto to 9 by 9 
#zero array. 

#Define the output image 
image_out = np.zeros((row,col)) 
#dynamic shifting of the flipped filter at each image coordinate and then 
#computing the convolved sum. 
for i in range(1, 1+row):
	for j in range(1, 1+col):
		arr_chunk = np.zeros((3,3)) 

		for k, k1 in zip(range(i-1, i+2), range(3)):
			for l, l1 in zip(range(j-1, j+2), range(3)):
				arr_chunk[k1, l1] = image1[k,l] 

		image_out[i-1, j-1] = np.sum(np.multiply(arr_chunk, filter_kernel_flipped)) 
print(image_out) 

#Based on the choice of image-processing filter, the nature of the output images will vary.
#for example, a Gaussian filter would create an output images that would be blurred version of the 
#input image, whereas a Sobel filter would detect the edges in an image and produce an output 
#image that contains the edges of the input image. 

##Common Image-processing filters:
##Mean filter code using openCV:
#The openCV module might give me the tools to finally finish the image processing exercise 
#in hands on machine learning that I was forced to put on hold. 

import cv2

img = cv2.imread("monalisa.jpg") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray") 
mean = 0
var = 100 
sigma = var**0.5
row, col = 225, 300
gauss = np.random.normal(mean, sigma, (row,col))
gauss = gauss.reshape(row, col) 
gray_noisy = gray + gauss 
plt.imshow(gray_noisy, cmap="gray")
plt.show()#Now I got it the only reason why the pyplot commands were not 
#working earlier is that I forget to use the function plt.show() to display 
#the plot and images with pyplot. 

#mean filter 
Hm = np.array([[1,1,1],[1,1,1],[1,1,1]])/float(9)
Gm = scipy.signal.convolve2d(gray_noisy,Hm,mode="same") 
plt.imshow(Gm, cmap="gray") 
plt.show() 

##Median filter using the opencv module again. 
#Look at pages 171 and 172 to see the salt and pepper noise methodology.
np.random.seed(0)
gray_sp = gray*1 
sp_indices = np.random.randint(0,21,[row,col]) 
for i in range(row):
	for j in range(col):
		if sp_indices[i,j]==0:
			gray_sp[i,j] = 0 
		if sp_indices[i,j] == 20:
			gray_sp[i,j] = 255
plt.imshow(gray_sp, cmap="gray")
plt.show() 

#medien filter code:
gray_sp_removed = cv2.medianBlur(gray_sp, 3)#this line of code actually 
#created the median filter automatically. Very useful. 
plt.imshow(gray_sp_removed, cmap="gray") 
plt.show()

gray_sp_removed_exp = gray*1
for i in range(row):
	for j in range(col):
		local_arr = [] 
		for k in range(np.max([0, i-1]), np.min([i+2,row])):
			for l in range(np.max([0,j-1]),np.min([j+2,col])):
				local_arr.append(gray_sp[k,l])
		gray_sp_removed_exp[i,j] = np.median(local_arr)
plt.imshow(gray_sp_removed_exp, cmap="gray") 
plt.show()
#Beautifully done will love to revisit the mathematics of this section to 
#see how the author came up with this solution. 

##Gaussian filter: The following lines create something called gaussian blurr 
Hg = np.zeros((20,20))
for i in range(20):
	for j in range(20):
		Hg[i,j] = np.exp(-((i-10)**2 + (j-10)**2)/10)
plt.imshow(Hg, cmap="gray") 
plt.show() 

gray_blur = scipy.signal.convolve2d(gray, Hg, mode="same") 
plt.imshow(gray_blur, cmap="gray") 
plt.show()

gray_high = gray - gray_blur 
plt.imshow(gray_high, cmap="gray") 
plt.show()    
gray_enhanced = gray + 0.025*gray_high
plt.imshow(gray_enhanced, cmap="gray") 
plt.show() 

##Gaussian based filters:
#page 174 horizontal filter kernel implemenation:
kernel1 = np.array([[0,0,0],[1,0,-1],[0,0,0]])
gray_hort1 = scipy.signal.convolve2d(gray, kernel1, mode="same") 
plt.imshow(gray_hort1, cmap="gray") 
plt.show() 

kernel2 = np.array([[0,1,0],[0,0,0],[0,-1,0]])
gray_vert1 = scipy.signal.convolve2d(gray, kernel2, mode="same") 
plt.imshow(gray_vert1, cmap="gray") 
plt.show() 
#Sweet I obtained the same outputs as the author. Check page 175.

##Sobel Edge-Detection filter:
#Convolution using a sobel filter:
#horizontal filter:
Hx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
Gx = scipy.signal.convolve2d(gray, Hx, mode="same") 
plt.imshow(Gx, cmap="gray") 
plt.show() 

Hy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32) 
Gy = scipy.signal.convolve2d(gray,Hy, mode="same") 
plt.imshow(Gy,cmap="gray") 
plt.show() 

G = (Gx*Gx + Gy*Gy)**0.5
plt.imshow(G, cmap="gray") 
plt.show()

##Identity transform filter:
ident_trans = np.array([[0,0,0],[0,1,0],[0,0,0]])
ident_array = scipy.signal.convolve2d(gray, ident_trans, mode="same") 
plt.imshow(ident_array, cmap="gray") 
plt.show() 

##Convolution layer: convolutional layer illustration using tensorflow: 
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x,W,strides=[1, strides, strides, 1], padding="SAME") 
	x = tf.nn.bias_add(x, b) 
	return tf.nn.relu(x)
#Interesting for this function (conv2d()) you don't need to filter kernel size. 
#will need to look into my older code, perhaps this is what I was messing up with. 

##Pooling layer example using tensorflow: 
def maxpool2d(x, stride=2):
	return tf.nn.max_pool(x,ksize=[1,stride,stride,1], strides=[1,stride,stride,1], padding="SAME") 

##convolutional Neural Network for digit Recognition on the MNIST Dataset: 
#The CNN takes in images of height 28, width 28, and depth 3 corresponding to the RGB channels. 
#The images go through the series of convolution, Relu activations, and max pooling operations twice 
#before being fed into the fully connected layer and finally to the output layer. The first convolutional 
#layer produces 64 feature maps, the second convolutional layer provides 128 feature maps, and the fully connected 
#layer has 1024 units. The max pooling layers have been chosen to reduce the feature map size by 1/4
#(which means a kernel filter of size 2 by 2 and a stride of 2). 

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

learning_rate = 0.01 
epochs = 1
batch_size = 256 
num_batches = int(np.ceil(int(X_train.shape[0]) / batch_size))
input_height = 28
input_width = 28 
dropout = 0.75
display_step = 1
filter_height = 5 
filter_width = 5
depth_in = 1
depth_out1 = 64
depth_out2 = 128 
n_classes = 10 

print(num_batches) 

tf.reset_default_graph()

x = tf.placeholder(tf.float32,[None, 28*28]) 
y = tf.placeholder(tf.float32,[None, n_classes]) 
keep_prob = tf.placeholder(tf.float32)#It seems that the author will implement 
#neuron drop by hand. 

weights = {
	"wc1": tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),
	"wc2": tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),
	"wd1": tf.Variable(tf.random_normal([int((input_height/4)*(input_height/4)*depth_out2),1024])),
	"out": tf.Variable(tf.random_normal([1024, n_classes]))
} 

biases = {
	"bc1": tf.Variable(tf.random_normal([64])),
	"bc2": tf.Variable(tf.random_normal([128])),
	"bd1": tf.Variable(tf.random_normal([1024])),
	"out": tf.Variable(tf.random_normal([n_classes]))
}

def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x,W,strides=[1, strides, strides, 1], padding="SAME") 
	x = tf.nn.bias_add(x, b) 
	return tf.nn.relu(x)

def maxpool2d(x, stride=2):
	return tf.nn.max_pool(x,ksize=[1,stride,stride,1], strides=[1,stride,stride,1], padding="SAME")

def conv_net(x, weights, biases, dropout):
	x = tf.reshape(x, shape=[-1, 28, 28, 1]) 
	#convolutional layer 1
	conv1 = conv2d(x, weights["wc1"],biases["bc1"]) 
	conv1 = maxpool2d(conv1, 2)

	#Convolutional layer 2
	conv2 = conv2d(conv1,weights["wc2"],biases["bc2"]) 
	conv2 = maxpool2d(conv2,2) 

	#fully connected layer
	fc1 = tf.reshape(conv2,[-1,weights["wd1"].get_shape().as_list()[0]]) 
	fc1 = tf.add(tf.matmul(fc1,weights["wd1"]),biases["bd1"]) 
	fc1 = tf.nn.relu(fc1)

	#apply dropout:
	fc1 = tf.nn.dropout(fc1,dropout) 

	#output class prediction:
	out = tf.add(tf.matmul(fc1,weights["out"]),biases["out"]) 
	return out

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(X_train), size = batch_size) 
	X_batch = X_train[indices] 
	y_batch = y_train[indices] 
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

	for i in range(10):
		a[i].imshow(np.reshape(X_test[i], (28, 28)))
		print(test_classes[i]) 

	plt.show() 







