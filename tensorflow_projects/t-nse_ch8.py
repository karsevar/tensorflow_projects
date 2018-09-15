from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_mldata 
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import LocallyLinearEmbedding

mnist = fetch_mldata("MNIST original")
X_mnist = mnist["data"] 
y_mnist = mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.2)
#this is going too slow. I will need to use the test set in place of the full 
#dataset. I hope this increases the computational speed a little. 

#this is for the pca method since the t-nse method was taking way too long.
pca = PCA(n_components = 2) 
X_reduced = pca.fit_transform(X_mnist)  
print(X_reduced) 

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y_mnist, cmap = plt.cm.magma)
plt.colorbar()
plt.show() 

#Locally Linear Embedding chaining together pca with lle:
pca = PCA(n_components = 154) #since 154 components was found to hold 
#95 percent of the variance in the dataset.
X_pca = pca.fit_transform(X_mnist) 
lle = LocallyLinearEmbedding(n_neighbors = 3, n_components = 2) 
X_reduced = lle.fit_transform(X_pca) 
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y_mnist, cmap = plt.cm.magma)
plt.colorbar()
plt.show() 


