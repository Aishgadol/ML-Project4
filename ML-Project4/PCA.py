from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def PCA_train(data, k):
    means=np.mean(data,axis=0)
    data=data-means
    scatter_mat=np.dot(data.T,data)
    eigenvalues, eigenvectors = np.linalg.eig(scatter_mat)
    E=eigenvectors[:,np.argsort(eigenvalues)[-k:]].transpose()
    y=np.matmul(E,data.transpose())
    return y, means, E
    # Download data to k dimensions

def PCA_test(test, mu, E):
    test=test-mu
    res=np.matmul(E,test.T)
    return res
def recover_PCA(data, mu, E):
    new_data=np.mul(np.linalg.inv(E),data.T)+mu
    return new_data