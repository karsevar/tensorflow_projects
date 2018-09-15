###Pro deep learning chapter 5 
##Markov Chain Monte Carlo sampling methods using the area of the 
#transcendental number Pi(pi). 
#Look at page 286 for more information:
import numpy as np 
number_sample = 100000
inner_area, outer_area = 0, 0 
for i in range(number_sample):
	x = np.random.uniform(0, 1) 
	y = np.random.uniform(0,1) 
	if (x**2 + y**2) < 1: 
		inner_area += 1 
	outer_area += 1 

print("the computed value of Pi:", 4 * (inner_area/float(outer_area)))
#Cool little program. 

##Bivariate Gaussian distribution through metropolis algorithm implementation:
import matplotlib.pyplot as plt 

#metropolis hastings algorithm:
import time 
start_time = time.time() 
#set up constants and initial variable conditions:
num_samples = 1000000
prob_density = 0 
#plan is to sample from a bivariate gaussian distribution with mean(0,0) 
#and covariance of 0.7 between the two variables.
mean = np.array([0,0]) 
cov = np.array([[1,0.7],[0.7,1]]) 
cov1 = np.matrix(cov) 
mean1 = np.matrix(mean) 
x_list, y_list = [], [] 
accepted_samples_count = 0 
#Normalizer of the probability distribution:
normalizer = np.sqrt(((2*np.pi)**2)*np.linalg.det(cov))
#start with initial point (0,0)
x_initial, y_initial = 0,0
x1, y1 = x_initial, y_initial 

for i in range(num_samples):
	#Set up the conditional probability distribution, taking the existing 
	#point as the mean and a small variance=0.2 so that points near the 
	#existing point have a high chance of getting sampled.
	mean_trans = np.array([x1, y1]) 
	cov_trans = np.array([[0.2,0],[0,0.2]]) 
	x2, y2 = np.random.multivariate_normal(mean_trans, cov_trans).T
	X = np.array([x2,y2]) 
	X2 = np.matrix(X) 
	X1 = np.matrix(mean_trans) 
	#Compute the probability density of the existing point and the new sampled 
	#point
	mahalnobis_dist2 = (X2 - mean1)*np.linalg.inv(cov)*(X2 - mean1).T 
	prob_density2 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist2)
	mahalnobis_dist1 = (X1 - mean1)*np.linalg.inv(cov)*(X1 - mean1).T
	prob_density1 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist1)
	acceptance_ratio = prob_density2[0,0] /float(prob_density1[0,0]) 

	if (acceptance_ratio >= 1) | ((acceptance_ratio < 1) and (acceptance_ratio >= np.random.uniform(0,1))):
		x_list.append(x2)
		y_list.append(y2) 
		x1 = x2
		y1 = y2 
		accepted_samples_count += 1 
end_time = time.time()

print("Time taken to sample " + str(accepted_samples_count) + " points ==> " + str(end_time - start_time)+ " seconds")
print("acceptance ratio ===> ", accepted_samples_count/float(100000))
#Time to display the samples generated 
plt.xlabel("X") 
plt.ylabel("y") 
plt.scatter(x_list, y_list, color = "black") 
plt.show() 
print("Mean of the sampled points") 
print(np.mean(x_list), np.mean(y_list))
print("Covariance matrix of the sample points") 
print(np.cov(x_list, y_list))
