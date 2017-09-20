import math
import sys
import numpy as np
import pandas as pd 
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split



plt.style.use('bmh') 
%matplotlib inline


def KnnDTW(object):
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d = lambda x, y: abs(x - y)):
        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxint * np.ones((M, N))

        # Initialise the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in xrange(1, M):
            cost[i, 0] = cost 

        
        




# kNN with cosine-similarity
def euclideanDist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(instance1[x] - instance2[x], 2)
    return math.sqrt(distance)


# kNN with sklearn instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fit the model using training set
knn.fit(X_train, y_train)

# predict the response on test set
pred = knn.predict(X_test)

# evaluate accuracy
print accuracy_score(y_test, pred)

# perform 10-fold cross validation
cv_scores = []

def knnClassify():
    for k in neighbours:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring = 'accuracy')  # cross_val_score a sklean built-in method
        cv_scores.append(scores.mean())

    

def main():
    pass
    k = 3
    neighbours = []

if __name__ == '__main__':
    main()
    # knnClassify()

