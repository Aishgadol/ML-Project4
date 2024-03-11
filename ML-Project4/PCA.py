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

x_train_new, mu, E = PCA_train(x_train_flatten, 81)
x_test_new = PCA_test(x_test_flatten, mu, E)
x_train_new_unflatten=x_train_new.reshape(len(x_train_new),9,9)
x_test_new_unflatten=x_test_new.reshape(len(x_test_new),9,9)

plt.subplot(131)
plt.title("me b4 hw4")
plt.imshow(x_train[5], cmap='gray')

plt.subplot(132)
plt.title("mw during hw4")
plt.imshow(x_train_new_unflatten[5].astype(float), cmap='gray')

plt.subplot(133)
plt.title("me after hw4")
plt.imshow(recover_PCA(x_train_new[5],mu,E).reshape(64,64).astype(float), cmap='gray')

plt.show()

x_train_flatten_zero_mean=x_train_flatten-np.mean(x_train_flatten,axis=0)
scatter=np.dot(x_train_flatten_zero_mean.T,x_train_flatten_zero_mean)
eigenvalues, eigenvectors = np.linalg.eig(scatter)

def EIG_CDF(eig_list):
	total_eig_sum=sum(eig_list)
	sorted_eigenvalues = np.sort(eig_list)[::-1]
	print(sorted_eigenvalues)
	eigenvalues_cumsum = np.cumsum(sorted_eigenvalues)
	eigenvalues_cumsum_normalized = eigenvalues_cumsum / eigenvalues_cumsum[-1]
	#find index where cumsum>=0.95
	amount = np.argmax(eigenvalues_cumsum_normalized >= 0.95) +1

	plt.plot(np.arange(1, len(sorted_eigenvalues)+1), eigenvalues_cumsum_normalized)
	plt.xlabel('Principal Component')
	plt.ylabel('Cumulative Proportion of Variance')
	plt.title(f'CDF of Eigenvalues - {amount} eigs preserves 95% of enetry')
	plt.show()

EIG_CDF(eigenvalues)

x_train_new_after, mu, E = PCA_train(x_train_flatten, 169)
x_test_new_after = PCA_test(x_test_flatten, mu, E)
x_train_new_unflatten_after=x_train_new_after.reshape(len(x_train_new_after),13,13)
x_test_new_unflatten_after=x_test_new_after.reshape(len(x_test_new_after),13,13)

plt.subplot(131)
plt.title("Original Image")
plt.imshow(x_train[5], cmap='gray')

plt.subplot(132)
plt.title("Image in lower dimension")
plt.imshow(x_train_new_unflatten_after[4].astype(float), cmap='gray')

plt.subplot(133)
plt.title("Recovered Image")
plt.imshow(recover_PCA(x_train_new_after[5],mu,E).reshape(64,64).astype(float), cmap='gray')

plt.show()
ks=range(1,20)
accs=[]
best_k=0
best_score_means=0
x_train_after_PCA, some_mu, some_e=PCA_train(x_train_flatten, 169)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(x_train_flatten,y_train)
    scores = cross_val_score(knn, x_train_after_PCA.astype(float), y_train, cv=5)  # 5-fold cross-validation
    accs.append(np.mean(scores))
    if(best_score_means<np.mean(scores)):
        best_score_means=np.mean(scores)
        best_k=k

plt.figure(figsize=(14,5))
plt.plot(ks, accs)
plt.xlabel('k')
plt.title(f'best k: {best_k}')
plt.xticks(ks)
plt.ylabel('avg accuracy')
plt.show()

knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train_new_after.astype(float), y_train)
y_pred=knn.predict(x_test_new_after.astype(float))
hits=0
for i in range(len(y_pred)):
    if(y_pred[i]==y_test[i]):
        hits+=1
acc=hits/len(y_pred)
print(f'acc on test is {acc}')