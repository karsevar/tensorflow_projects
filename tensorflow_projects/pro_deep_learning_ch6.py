###Pro deep learning chapter 6 Advanced Neural networks: 
##Image segmentation using an arbitrary pixel intensity threshold 
#Used for subject and background identification.
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

img = cv2.imread("monalisa.jpg") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray, cmap="gray") 
plt.show()
row, col = np.shape(gray) 
gray_flat = np.reshape(gray, (row*col, 1))[:, 0] 
ax = plt.subplot(222) 
ax.hist(gray_flat, color="gray")
plt.show()  
gray_const = []

#The author interestingly picks 150 as the threshold. Will need to see how 
#this works out.
#Changing the threshold to 100 results in an image with a better background 
#rendition.
for i in range(len(gray_flat)):
	if gray_flat[i] < 100:
		gray_const.append(255) 
	else:
		gray_const.append(0) 
#Creating a segmentation logic.
gray_const = np.reshape(np.array(gray_const), (row, col))
bx = plt.subplot(333) 
bx.imshow(gray_const, cmap="gray") 
plt.show()#Interesting!!! the threshold did get the subject of the 
#image and part of the background will need to see what the ostu's method 
#does with this image. 

##Otsu's thresholding method: 
hist_dist = 256*[0] 
#compute the frequence count of each of the pixels in the image 
for i in range(row):
	for j in range(col):
		hist_dist[gray[i,j]] += 1
#Normalize the frequencies to produce probabilities
hist_dist = [c/float(row*col) for c in hist_dist] 
plt.plot(hist_dist) 
plt.show() 
#Compute the between segment variance:
def var_c1_c2_func(hist_dist, t):
	u1, u2, p1, p2, u = 0,0,0,0,0
	for i in range(t+1):
		u1 += hist_dist[i]*i 
		p1 += hist_dist[i] 
	for i in range(t+1, 256):
		u2 += hist_dist[i]*i 
		p2 += hist_dist[i] 
	for i in range(256):
		u += hist_dist[i]*i 
	var_c1_c2 = p1*(u1-u)**2 + p2*(u2 - u)**2 
	return var_c1_c2 
#Iteratively run through all the pixel intensities from 0 to 255 and 
#choose the one that maximizes the variance 

variance_list = [] 
for i in range(256):
	var_c1_c2 = var_c1_c2_func(hist_dist, i) 
	variance_list.append(var_c1_c2) 
#Fetch the threshold that maximizes the variance 
t_hat = np.argmax(variance_list)
print(t_hat)#That actual threshold was 89 according to the algorithm. 

#compute the segmented image based on the threshold t_hat 
gray_recons = np.zeros((row, col)) 
for i in range(row):
	for j in range(col):
		if gray[i,j] <= t_hat:
			gray_recons[i,j] = 255
		else:
			gray_recons[i,j] = 0 
plt.imshow(gray_recons, cmap="gray") 
plt.show() 
#Why better than the arbitrary threshold method. I can actually 
#see some aspects within the background. 




