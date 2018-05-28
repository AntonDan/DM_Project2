from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from ast import literal_eval
from random import randint
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import numpy as np

import math

def haversine(X, Y):
    lat1, lon1 = X 
    lat2, lon2 = Y
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d 

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

def lcs_length(a, b):
    table = [[([],0)] * (len(b) + 1) for _ in xrange(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            if haversine(ca, cb) < 0.2:
                path, length = table[i - 1][j - 1]
                table[i][j] = (path + [ca], length + 1)
            else:
                patha, lengtha = table[i][j - 1]
                pathb, lengthb = table[i - 1][j]
                if lengtha > lengthb:
                    table[i][j] = (patha, lengtha)
                else:
                    table[i][j] = (pathb, lengthb)
    path, length = table[-1][-1]
    return path, length

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



# MAIN
testSet = pd.read_csv(
	'test_set_a2.csv',
	converters={"Trajectory": literal_eval},
    delimiter='\t'    
)

trainSet = pd.read_csv(
	'train_set.csv', # replace with the correct path
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)

trajectories = trainSet["Trajectory"]
journeyPatternIds = trainSet['journeyPatternId']

test_trajectories = testSet["Trajectory"]

samples = []
for i in range(0, len(trajectories.keys())):
    index = trajectories.keys()[i]                                # Get a random applicable index on the dataset
    samples.append([])
    for t, lon, lat in trajectories[index]:                         # Fill them with the desired trip info
        samples[i].append([lon, lat])
#samples=samples[1:100]

test = []
for i in range(0, len(test_trajectories.keys())):
    index = test_trajectories.keys()[i]                                # Get a random applicable index on the dataset
    test.append([])
    for t, lon, lat in test_trajectories[index]:                         # Fill them with the desired trip info
        test[i].append([lon, lat])

for t in range(0, len(test)):
    amount = 0
    minimum = 0
    top = []
    test_index = 0
    print ("Test " , t)
    for i in range(0,len(samples)):
        path, length = lcs_length(samples[i], test[t])
        if (length > minimum):
    #     dist, cost, acc, path = fastdtw(samples[i], test[0], haversine)
            if (len(top) < 5):
                top += [(trajectories.keys(), path, length)]
            else:
                for i in range(0, 5):
                    _, _,   _length = top[i]
                    if _length == minimum:
                        top[i] = (trajectories.keys(), path, length)
                        break
                minimum = inf
                for _, _, _length in top:
                    if _length < minimum:
                        minimum = _length
    print top
# vizualize
# from matplotlib import pyplot as plt
# plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
# plt.plot(path[0], path[1], '-o') # relation
# plt.xticks(range(len(x)), x)
# plt.yticks(range(len(y)), y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('tight')
# plt.title('Minimum distance: {}'.format(dist))
# plt.show()