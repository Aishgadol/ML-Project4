import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Kmeans:

	def __init__(self, n_clusters, max_iter=100, random_state=123):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.random_state = random_state

	def initialize_centroids(self, X):
		np.random.RandomState(self.random_state)
		random_idx = np.random.permutation(X.shape[0])
		centroids = X[random_idx[:self.n_clusters]]
		return centroids

	def reassign_centroids(self, X, labels):
		centroids = np.zeros((self.n_clusters, X.shape[1]))
		# Implement here
		for i in range(self.n_clusters):
			centroids[i]=np.mean(X[labels==i],axis=0)
		return centroids

	def compute_distance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
			distance[:, k] = np.square(row_norm)
		return distance

	def find_closest_cluster(self, distance):
		return np.argmin(distance, axis=1)

	def compute_sse(self, X, labels, centroids):
		distance = np.zeros(X.shape[0])
		for k in range(self.n_clusters):
			distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
		return np.sum(np.square(distance))

	def fit(self, X):
		self.centroids = self.initialize_centroids(X)
		for i in range(self.max_iter):
			old_centroids = self.centroids
			# For each point, calculate distance to all k clustes.
			distances=self.compute_distance(X,old_centroids)
			self.labels =np.argmin(distances,axis=1) # Assign the labels with closest distance' cluster.
			self.centroids = self.reassign_centroids(X, self.labels)# Update the centroids
			if np.all(old_centroids == self.centroids):
				break
		self.error = self.compute_sse(X, self.labels, self.centroids)

	def predict(self, X):
		distance = self.compute_distance(X, self.centroids)
		return self.find_closest_cluster(distance)

df = pd.read_csv('exams.csv').to_numpy()
data=df[:,:-1]
labels=df[:,-1]
data_label0=data[labels==0]
data_label1=data[labels==1]
plt.figure(figsize=(8, 6))
plt.scatter(data_label0[:, 0], data_label0[:, 1], color='blue', label='Label 0')
plt.scatter(data_label1[:, 0], data_label1[:, 1], color='red', label='Label 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Data with Labels')
plt.legend()
plt.grid(False)
plt.show()

clust=Kmeans(n_clusters=2)
clust.fit(data)
labels = clust.labels
centroids = clust.centroids
for i in range(len(centroids)):
    print(f'centroid {i} is at: {centroids[i]}')
c0 = data[labels == 0]
c1 = data[labels == 1]

plt.scatter(c0[:,0], c0[:,1], c='green', label='cluster 1')
plt.scatter(c1[:,0], c1[:,1], c='blue', label='cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='centroid')
#plt.legend()

plt.show()

