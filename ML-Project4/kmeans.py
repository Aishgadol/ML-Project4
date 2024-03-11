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
'''
using elbow method to choose another number of centroids, between 1 and 10
elbow method means that we calc the SSE (loss) per number of clusters, and choose where the loss betweeen
i and i+1 clusters is minimal, then i is the optimal number of clusters
'''
sse = []
list_k = list(range(1,11))
for k in list_k:
    clust=Kmeans(n_clusters=k)
    clust.fit(data)
    labels = clust.labels
    centroids = clust.centroids
    sse.append(clust.error)
print(sse)
best_k=0
best_loss_difference=sse[0]
for i in range(len(sse)-1):
    if(sse[i]-sse[i+1]<best_loss_difference and sse[i]-sse[i+1]>0 and i<6):
        best_loss_difference=sse[i]-sse[i+1]
        best_k=i+1
print(f'best k: {best_k} with difference in sse: {best_loss_difference}')
'''Plot sse against k'''
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.show()

'''
now we plot the clustered data according to the best k we found
we'll add convexhull to draw lines around each cluster so it's more fun to look at 
'''
from scipy.spatial import ConvexHull
colors=['red', 'blue', 'green', 'magenta', 'cyan', 'black','pink','yellow','orange']
clust = Kmeans(n_clusters=best_k)
clust.fit(data)
labels = clust.labels
centroids = clust.centroids
for i in range(len(centroids)):
	print(f'centroid {i} is at: {centroids[i]}')
clusters={i:data[labels==i] for i in range(best_k)}
for i in range(best_k):
	plt.scatter(clusters[i][:,0],clusters[i][:,1], c=colors[i],label=f'cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='centroid')
plt.legend(fontsize='5')
for i in range(len(centroids)):
	# Find the convex hull of the cluster points
	cluster_points = data[labels == i]
	hull = ConvexHull(cluster_points)

	# Plot the convex hull
	for simplex in hull.simplices:
		plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], c='gray')

plt.show()