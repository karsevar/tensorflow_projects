###Hands on machine learning with Scikit learning and tensorflow
###chapter 13 exercises part 2:

##8.) Sadly, I'm not comfortable enough with convolutional neural networks 
#to tackle this problem in a reasonable time frame. In fact, I still can't
#got my mnist data convolutional neural network to work properly with only 
#two convolutional layers and two pooling average layers. 

import tensorflow as tf
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg 
import numpy as np 
import os 

width = 299 
height = 299 
channels = 3

test_image = mpimg.imread("saint3.jpeg")[:, :, :channels]
#Sweet jpegs work perfectly with this importing method. Remember that the images 
#are saved in the pythonworks directory. Will need to clean this up later. 
plt.imshow(test_image) 
plt.axis("off") 
plt.show()

test_image = 2 * test_image - 1 
print(test_image.shape) 

import sys 
import tarfile 
from six.moves import urllib 

TF_MODELS_URL = "http://download.tensorflow.org/models" 
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz" 
INCEPTION_PATH = os.path.join("datasets","inception") 
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt") 

def download_progress(count, block_size, total_size):
	percent = count * block_size * 100 // total_size 
	sys.stdout.write("\rDownloading: {}%".format(percent))
	sys.stdout.flush() 

def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path = INCEPTION_PATH):
	if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
		return 
	os.makedirs(path, exist_ok=True)
	tgz_path = os.path.join(path, "inception_v3.tgz") 
	urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress) 
	inception_tgz = tarfile.open(tgz_path) 
	inception_tgz.extractall(path=path) 
	inception_tgz.close() 
	os.remove(tgz_path)

fetch_pretrained_inception_v3() 

import re
import urllib3 
import pandas as pd 

CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(x*)\s*$", re.M | re.U) 

def load_class_names():
	data = pd.read_csv("imagenet_class_names.txt", sep=" ", header = None) 
	return data[2].tolist()     
 
class_names = ["background"] + load_class_names()
print(class_names) 

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim 

tf.reset_default_graph() 

X = tf.placeholder(tf.float32, shape = [None, 299, 299, 3], name = "X") 
with slim.arg_scope(inception.inception_v3_arg_scope()):
	logits, end_points = inception.inception_v3(
		X, num_classes = 1001, is_training=False) 
predictions = end_points["Predictions"] 
saver = tf.train.Saver() 

def prepare_image(image, target_width = 299, target_height = 299):
	"""crops the image for data augmentation."""
	image = imresize(image, (target_width, target_height)) 
	return image.astype(np.float32)

X_test = test_image.reshape(height, width, channels) 

#this is the image 
#of the newfoundland from earlier in the sublime session. 

with tf.Session() as sess:
	saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH) 
	predictions_val = predictions.eval(feed_dict={X: X_test})





