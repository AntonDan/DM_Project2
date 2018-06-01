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

from gmplot import gmplot
from timeit import default_timer as timer

import math
from haversine import haversine

# LCSS function
# Input: a[1...m], b[1...n]
# Returns: the longest common subsequence and its length
# Creates a 2 dimensional matrix table[1...m,1...n] and initializes it
# for every coordinate in a then for every coordinate in j 
# if those coordinates represent points within 200 meteres
# update the respective table element using the previous (diagonally) element info
# else just pass on the top or left element based on which has the largest length   
def lcss(a, b):
    table = [[([],0)] * (len(b) + 1) for _ in range(len(a) + 1)]
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

# MAIN
testSet = pd.read_csv(
	'../test_set_a2.csv', # replace with the correct path
	converters={"Trajectory": literal_eval},
    delimiter='\t'    
)

trainSet = pd.read_csv(
	'../train_set.csv', # replace with the correct path
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
        samples[i].append([lat, lon])

test = []
for i in range(0, len(test_trajectories.keys())):
    index = test_trajectories.keys()[i]                                # Get a random applicable index on the dataset
    test.append([])
    for t, lon, lat in test_trajectories[index]:                         # Fill them with the desired trip info
        test[i].append([lat, lon])

for t in range(0, len(test)):
    amount = 0
    minimum = 0
    top = []
    test_index = 0
    start_time = timer()
    print ("Test " , t)
    for i in range(0,len(samples)):
        path, length = lcss(samples[i], test[t])
        if (length > minimum):
            if (len(top) < 5):
                top += [(trajectories.keys()[i], path, length)]
            else:
                for j in range(0, 5):
                    _, _,   _length = top[j]
                    if _length == minimum:
                        top[j] = (trajectories.keys()[i], path, length)
                        break
                minimum = inf
                for _, _, _length in top:
                    if _length < minimum:
                        minimum = _length
    
    delta = timer() - start_time
    print("Test Trip " + str(t+1) + " time_delta: " + str(delta) + " seconds")

    gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)

    longitudes = list()
    latitudes = list()
    for lat, lon in test[t]:
        longitudes.append(lon)
        latitudes.append(lat)

    gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map
    gmap.draw("Test Trip " + str(t+1) + ".html")

    for j, (key, path, length) in enumerate(top):

        longitudes = list()
        latitudes = list()
        for _, lon, lat in trajectories[key]:
            longitudes.append(lon)
            latitudes.append(lat)
        gmap.plot(latitudes, longitudes, "green", edge_width=2)		# Plot it on the map

        longitudes = list()
        latitudes = list()
        for lat, lon in path:
            longitudes.append(lon)
            latitudes.append(lat)
        gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map

        print("Test Trip " + str(t+1) + " Neighbor " + str(j+1) + " j_id: " + str(journeyPatternIds[key]) + " matching_points: " + str(length))
        gmap.draw("TT " + str(t+1) + " Neighbor " + str(j+1) + ".html")