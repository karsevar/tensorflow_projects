###Pro Deep learning chapter 3 VGG16 pre-trained model which will be 
#used to classify dog and cat photographs. 
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from scipy.misc import imresize 
from sklearn.model_selection import train_test_split 
import cv2 
import tensorflow.contrib.slim as slim 
from tensorflow.contrib.slim.nets import vgg 
from tensorflow.contrib.slim.preprocessing import vgg_preprocessing 
from mlxtend.preprocessing import shuffle_arrays_unison 






