from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd
from ast import literal_eval
from random import randint
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import numpy as np
from gmplot import gmplot

from knn import KNeighborsClassifier

trainSet = pd.read_csv(
	'../train_set.csv', # replace with the correct path
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)

trajectories = trainSet["Trajectory"]
journeyPatternIds = trainSet['journeyPatternId']

testSet = pd.read_csv(
	'../test_set_a1.csv', # replace with the correct path
    sep='\t',
    converters={"Trajectory": literal_eval}
)
testSet = testSet["Trajectory"]

samples = []
for i in range(0, len(trajectories.keys())):
    index = trajectories.keys()[i]                                # Get a random applicable index on the dataset
    samples.append([])
    for t, lon, lat in trajectories[index]:                         # Fill them with the desired trip info
        samples[i].append([lat, lon])

test = []
for i in range(0, len(testSet.keys())):
    index = testSet.keys()[i]                                # Get a random applicable index on the dataset
    test.append([])
    for t, lon, lat in testSet[index]:                         # Fill them with the desired trip info
        test[i].append([lat, lon])

samples = samples[:10]
journeyPatternIds = journeyPatternIds[:10]
test = test[:1]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(samples, journeyPatternIds)

result = neigh.kneighbors(test, 1)

gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)

longitudes = list()
latitudes = list()
for lat, lon in test[0]:
    longitudes.append(lon)
    latitudes.append(lat)

gmap.plot(latitudes, longitudes, "black", edge_width=2)		# Plot it on the map

longitudes = list()
latitudes = list()
for lat, lon in result[0]:
    longitudes.append(lon)
    latitudes.append(lat)

gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map

gmap.draw("eyetest.html")