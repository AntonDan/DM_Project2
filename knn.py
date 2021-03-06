import numpy as np
import itertools
import operator

from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import DistanceMetric

from sklearn.utils.estimator_checks import check_estimator

# Our custom K-Nearest Neighbor implementation
class KNeighborsClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors=5, n_jobs=1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y):

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store arguments, since we have a lazy student type classifier
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X, dist):
        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        predicted_labels = list()

        for qitem in X:
            neighbors_distance = list()
            for idx, fitem in enumerate(self.X_):
                neighbors_distance.append( (idx, fastdtw(fitem, qitem, dist)) )

            sorted_neighbors_distance = sorted(neighbors_distance, key=lambda neighbors_distance: neighbors_distance[1])
            predicted_labels.append(self.majority_voting(list(itertools.islice(sorted_neighbors_distance, self.n_neighbors))))

        return np.array(predicted_labels)

    # Majority voting mechanism with distance as weight
    def majority_voting(self, neighbors):
        majority_dict = dict()

        for neighbor_index, distance in neighbors:
            if self.y_[neighbor_index] in majority_dict:
                majority_dict[self.y_[neighbor_index]] += distance
            else:
                majority_dict[self.y_[neighbor_index]] = distance

        return min(majority_dict.items(), key=operator.itemgetter(1))[0]

    # Needs to return the k best neighbors (we need those with the least distance)
    def kneighbors(self, X, dist, n_neighbors=5):
        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        k_neighbors = list()

        for qitem in X:
            neighbors_distance = list()
            for idx, fitem in enumerate(self.X_):
                neighbors_distance.append( (idx, fastdtw(fitem, qitem, dist)) )

            sorted_neighbors_distance = sorted(neighbors_distance, key=lambda neighbors_distance: neighbors_distance[1])
            k_neighbors.append(list(itertools.islice(sorted_neighbors_distance, n_neighbors)))

        k_neighbors_actual = list()
        for item in k_neighbors:
            temp = list()
            for neighbor in item:
                temp.append( (self.y_[neighbor[0]], neighbor[1][0], self.X_[neighbor[0]]) )
            if len(temp) > 0:
                k_neighbors_actual.append(temp)

        return k_neighbors_actual

def dtw(x, y, dist=euclidean_distances):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array 
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = dist(x, y)
    print(D1)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)