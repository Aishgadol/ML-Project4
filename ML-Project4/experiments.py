from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def plotty(data,centered_data):
    plt.figure(figsize=(10, 8))
    plt.title('Random Dataset')
    plt.xlabel('Features')
    plt.ylabel('Values')
    #plt.axhline(0, color='black', linewidth=1.0)
    #plt.axvline(0, color='black', linewidth=1.0)
    plt.scatter(data[:, 0], data[:, 1], label='Samples', color='blue')
    plt.scatter(data[:, 0], data[:, 1], label='Original Data', color='red')

    # Plotting centered data in blue
    plt.scatter(centered_data[:, 0], centered_data[:, 1], label='Centered Data', color='blue')
    plt.legend()
    plt.grid(True)
    plt.show()
data= np.random.randint(1,100,(30, 5))
print(data)
means=np.mean(data,axis=0)
centerd_data=data-means
print(centerd_data)
print(np.mean(data,axis=0))
print(np.mean(centerd_data,axis=0))
scatter=np.dot(centerd_data.T,centerd_data)
print(scatter.shape)
eigenvalues,eigenvectors=np.linalg.eig(scatter)
#print(eigenvectors)
#print(np.argsort(eigenvalues))
print(data.shape)
E=eigenvectors[:,np.argsort(eigenvalues)[-2:]].T
print(E.shape)
y=np.matmul(E,centerd_data.T).T
print(y)