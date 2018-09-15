###Hands on machine learning with scikit learn and tensorflow:
###Chapter 8 Dimensionality Reduction:
##Two main approaches to dimensionality rediction (projection and manifold)
#learning), and three of the most popular dimension reduction techniques are 
#PCA, kernel PCA, and LLE.

##Main Approaches for dimensionality reduction:
##Projection:
#Look at the swiss roll toy dataset articles and problems. This is 
#found on page 209.

##Manifold Learning:
#Look at page 209 and 210 to see the descriptions of this constraint.

##PCA: principle component analysis is by far the most popular dimensionality 
#reduction algorithm. First it identifies the hyperplane that lies closest 
#to the data and then it projects the data onto it.

#Interesting the author finds the principle components within a training 
#set through singular value decomposition.
#This can be computed through the numpy module's linalg.svd() function.
import numpy as np 

np.random.seed(4) 
m = 60 
w1, w2 = 0.1, 0.3 
noise = 0.1 

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) /2 
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m) 

#Calculating the principle components:
X_centered = X - X.mean(axis = 0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:,  0]
c2 = Vt.T[:, 1] 
print(c1, c2)#Interesting so the objects c1 and c2 are both principle components 
#will need to play around with this a little more.

##Projecting down to d dimensions:
#Once you have identified all the principal components, you can reduce 
#the dimensionality of the dataset down to d dimensions by projecting it 
#onto the hyperplane defined by the firest d principal components. 

#to project the training set onto the hyperplane, you can simply compute 
#the dot product of the training set matrix X by the matrix W_d (which is the 
#first principal component), defined as the matrix containing the first 
#principal components.

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
print(W2)
print(X2D)

##Using Scikit learn:
#Scikit-learn's PCA class implements PCA using SVD decomposition.
from sklearn.decomposition import PCA 

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)#This function reduces the dimensions down 
#to a specified level (which is dictated by the n_components argument).

print(pca.components_.T[:, 0])#First principal component
print(pca.components_.T[:, 1])#Second principal component 

##Explained Variance Ratio:
#Another very useful piece of information is the explained variance ratio 
#of each principal component available via the explained_variance_ratio_ 
#variable. It indicates the percentage of variance that lays within each of the 
#axes. 
print(pca.explained_variance_ratio_)

##Choosing the right number of dimensions:
#The following code computes PCA without reducing dimensionality, then 
#computes the minimum number of dimensions required to preserve 95 percent of the training
#set's variance:
from sklearn.model_selection import train_test_split 

X_train, X_test = train_test_split(X, test_size = 0.15)
pca = PCA() 
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 

#Interesting notes on the function: You could set n_components = d and 
#run the PCA again or you can just specify the percentage of variance 
#you want to preserve.

pca = PCA (n_components=0.95)
X_reduced = pca.fit_transform(X_train) 

#plot visualization of the explained variance:
import matplotlib.pyplot as plt 
plt.plot(range(0, 3), cumsum)
plt.show()#Not a very good plot, but this is the best I can do with 
#the data.

##PCA for compression:
from six.moves import urllib 
from sklearn.datasets import fetch_mldata 

mnist = fetch_mldata("MNIST original")
X_mnist = mnist["data"]
y_mnist = mnist["target"] 
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, train_size = 0.8)
print(X_train.shape)
#pca = PCA(n_components=0.95) 
#X_reduced = pca.fit_transform(X_train) 
#I can't seem to display the reduced dimensions onto the console through
#this method. Will need to think of another way to go about this.

pca = PCA() 
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 
print(d)# This dataset was reduced to a total of 153 dimensions. Very interesting.

#The mean squared distance between the original data and the reconstructed data
#is called the reconstruction error. You can reconstruct the original dataset through
#the function inverse_transform() 

pca = PCA(n_components = 153) 
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

#plt.plot(range(0, 784), cumsum) 
#plt.show()#the perfect elbow plot visualization. Most of the variance is 
#explained by 100 and 200 dimensions. 

##Incremental PCA:
#the following code splits the MNIST dataset into 100 mini batches (using 
#Numpy's array_split() function) and feeds them to scikit learn's incrementalPCa 
#class to reduce the dimensionality of the MNIST dataset down to 154 dimensions 

from sklearn.decomposition import IncrementalPCA

#n_batches = 100
#inc_pca = IncrementalPCA(n_components = 153) 
#for X_batch in np.array_split(X_train, n_batches):
	#inc_pca.partial_fit(X_batch) 

#X_reduced = inc_pca.transform(X_train) 

#You can also use the memmap class to do this same problem:
#look at page 217.

##Randomized PCA:
rnd_pca = PCA(n_components = 153, svd_solver = "randomized")
X_reduced = rnd_pca.fit_transform(X_train) 

##Kernel PCA:
#It seems that this instance has the same charactistics as the support vector machine 
#methods from the earlier chapters.
from sklearn.decomposition import KernelPCA 

rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 0.04)
X_reduced = rbf_pca.fit(X) #Interesting the console says that 
#kernelPCA doesn't have a transform_fit() method. Will need to look 
#into this. 

##Selecting a kernel and tuning hyperparameters.
#To find the best hyperparameters for this method (which is a unsupervised 
#statistical training method) you can use the function GridSearchCV().

from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons

moons = make_moons(n_samples = 1000, shuffle = True, noise = 0.2, random_state = 42)
X = moons[0]
y = moons[1] 

clf = Pipeline([
	("kpca", KernelPCA(n_components = 2)),
	("log_reg", LogisticRegression())
])

param_grid = [{
	"kpca__gamma": np.linspace(0.03, 0.05, 10),
	"kpca__kernel": ["rbf", "sigmoid"] 
}]
grid_search = GridSearchCV(clf, param_grid, cv = 3) 
grid_search.fit(X, y) 
print(grid_search.best_params_)
#cool!! This works perfectly. I was forced to use the make_moons 
#dataset generator. 

#to assess which hyperparameter is the most effective, you need to use the 
#reconstruction pre-image error rate. This rate will need to be reduced to the 
#minimum before a model can be considered statistically accurate. 

rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 0.0478,
	fit_inverse_transform = True) 
X_reduced = rbf_pca.fit_transform(X) 
X_preimage = rbf_pca.inverse_transform(X_reduced) 

from sklearn.metrics import mean_squared_error
print(mean_squared_error(X, X_preimage))#pre_image error rate looks very 
#low, which means that this model is prefect for the dataset. I wonder 
#what happens if I increase the noise parameter within the make_moons() 
#dataset. 

##LLE 
#Locally Linear Embedding is another very powerful non-linear dimensionality 
#reduction technique. It is a Manifold Learning technique that does not rely on
#projections but instead it looks for a low dimensional representation 
#of the training set where each training instance linearly relates 
#to its closest neighbors 

from sklearn.manifold import LocallyLinearEmbedding 

lle = LocallyLinearEmbedding(n_components = 2, n_neighbors = 10)
X_reduced = lle.fit_transform(X)  







