###Pro deep learning chapter 6 
##Watershed algorithm tensorflow implemenation 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import ndimage 
from skimage.feature import peak_local_max 
from skimage.morphology import watershed 

im = cv2.imread("car_tree .jpg")
print(np.shape(im))
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
plt.imshow(imgray, cmap="gray") 
plt.show() 
#Threshold the image to convert it to binary image based on otsu's method 
thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
im2, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
y = cv2.drawContours(imgray, contours, -1, (0,255,0), 3) 
#Relabel the thresholded image to be consisting of only 0 and 1 
#As the input image to distance_transform_edt should be in this format.
thresh[thresh==255] = 5
thresh[thresh==0] = 1
thresh[thresh==5] = 0 

#The distance transform edt and the peak local max functions help building the markers 
#by detecting the points near the center points of the car and tree.
D = ndimage.distance_transform_edt(thresh) 
localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh) 
markers = ndimage.label(localMax, structure=np.ones((3,3)))[0] 
#Provide the EDT distance matrix and the markers to the watershed 
#algorithm to detect the cluster's labels for each pixel.
labels = watershed(-D, markers, mask=thresh) 
print("[INFO] {} unique segments found".format(len(np.unique(labels)) -1))
#Create the contours for each label and append to the plot 
for k in np.unique(labels):
	if k != 0:
		labels_new = labels.copy() 
		labels_new[labels == k] = 255 
		labels_new[labels != k] = 0 
		labels_new = np.array(labels_new, dtype="uint8") 
		im2, contours, hierarchy = cv2.findContours(labels_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
		z = cv2.drawContours(imgray, contours, -1, (0,255,0), 3) 
		plt.imshow(z, cmap="gray") 
plt.show()
#Interesting the image segmenation algorithm look more at the tree 
#and the road. I believe that this implemenation will still have a 
#hard time differentiating between the road, tree, and the car.  

##Image segmentation using k-means with Tensorflow:
np.random.seed(0) 
img = cv2.imread("car_tree .jpg") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray, cmap="gray")
plt.show() 
row, col, depth = img.shape 
#Collapse the row and column axis for faster matrix operation 
img_new = np.zeros(shape=(row*col, 3))
glob_ind = 0 
for i in range(row):
	for j in range(col):
		u = np.array([img[i,j,0], img[i,j,1],img[i,j,2]])
		img_new[glob_ind, :] = u 
		glob_ind += 1 

K = 3#A total of three defined centroids 
num_iter = 20 
for g in range(num_iter):
#Define cluster for storing the cluster number and out_dist to store the distances
#from centroid.
	clusters = np.zeros((row*col, 1)) 
	out_dist = np.zeros((row*col, K))
	centroids = np.random.randint(0,255,size=(K,3))
	for k in range(K):
		diff = img_new - centroids[k,:]
		diff_dist = np.linalg.norm(diff, axis=1) 
		out_dist[:, k]= diff_dist 
	#Assign the cluster with minimum distance to a pixel location 
	clusters = np.argmin(out_dist, axis=1) 
	#REcompute the clusters 
	for k1 in np.unique(clusters):
		centroids[k1,:] = np.sum(img_new[clusters == k1, :], axis=0)/np.sum([clusters == k1]) 
	#REshape the cluster labels in two-dimensional image form 
	clusters = np.reshape(clusters, (row,col))
	out_image = np.zeros(img.shape) 
	#Form the 3d image with the labels replaced by their corresponding centroid pixel 
	for i in range(row):
		for j in range(col):
			out_image[i, j, 0] = centroids[clusters[i,j],0]
			out_image[i,j,1] = centroids[clusters[i,j],1] 
			out_image[i,j,2] = centroids[clusters[i,j],2]

out_image = np.array(out_image, dtype="uint8") 
plt.imshow(out_image)
plt.show() 



