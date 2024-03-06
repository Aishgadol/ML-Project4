from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def PCA_train(data, k):
    means=np.mean(data,axis=0)
    
	# Download data to k dimensions

def PCA_test(test, mu, E):
	# Implement here

def recover_PCA(data, mu, E):
	# Implement here