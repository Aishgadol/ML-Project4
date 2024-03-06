from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

data= np.random.randint(1, 10, size=(4, 5))
print(data)
print(np.mean(data,axis=0))