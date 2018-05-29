from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd
from ast import literal_eval
from random import randint
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from gmplot import gmplot

from knn import KNeighborsClassifier

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

# trainSet = pd.read_csv(
# 	'../train_set.csv', # replace with the correct path
# 	converters={"Trajectory": literal_eval},
# 	index_col='tripId'
# )

# trajectories = trainSet["Trajectory"]
# journeyPatternIds = trainSet['journeyPatternId']

# testSet = pd.read_csv(
# 	'../test_set_a1.csv', # replace with the correct path
#     sep='\t',
#     converters={"Trajectory": literal_eval}
# )
# testSet = testSet["Trajectory"]

# testSet2 = pd.read_csv(
# 	'../test_set_a2.csv', # replace with the correct path
#     sep='\t',
#     converters={"Trajectory": literal_eval}
# )
# testSet2 = testSet2["Trajectory"]

# samples = []
# for i in range(0, len(trajectories.keys())):
#     index = trajectories.keys()[i]
#     samples.append([])
#     for t, lon, lat in trajectories[index]:
#         samples[i].append([lat, lon])

# labels = []
# for i in range(0, len(journeyPatternIds.keys())):
#     index = journeyPatternIds.keys()[i]
#     labels.append(journeyPatternIds[index])

# test = []
# for i in range(0, len(testSet.keys())):
#     index = testSet.keys()[i]
#     test.append([])
#     for t, lon, lat in testSet[index]:
#         test[i].append([lat, lon])

# test2 = []
# for i in range(0, len(testSet2.keys())):
#     index = testSet2.keys()[i]
#     test2.append([])
#     for t, lon, lat in testSet2[index]:
#         test2[i].append([lat, lon])

# maximum = int(abs(len(samples) / 10))
# samples = samples[:maximum]
# labels = labels[:maximum]
# test = test[:2]
# test2 = test2[:1]

# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(samples, labels)

# result = neigh.kneighbors(test, haversine, 5) # a1 question
# result = neigh.predict(test2, haversine)    # 3rd question

# 3rd question
# dic = {
#     "Test_Trip_ID" : range(len(test2)),
#     "Predicted_JourneyPatternID" : result
# }
# out_df = pd.DataFrame(dic, columns=['Test_Trip_ID', 'Predicted_JourneyPatternID'])
# out_df.to_csv("testSet_JourneyPatternIDs.csv", sep='\t', index=False)

# a1 question
# for i, answer in enumerate(result):
#     gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)

#     longitudes = list()
#     latitudes = list()
#     for lat, lon in test[i]:
#         longitudes.append(lon)
#         latitudes.append(lat)

#     gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map
#     gmap.draw("Test Trip " + str(i+1) + ".html")

#     for j, (journey_id, dtw_distance, trajectory) in enumerate(answer):

#         # print ("journey_id: ", journey_id, " dtw_distance: ", dtw_distance)
#         # print (trajectory)

#         longitudes = list()
#         latitudes = list()
#         for lat, lon in trajectory:
#             longitudes.append(lon)
#             latitudes.append(lat)

#         gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map
#         print("Test Trip " + str(i+1) + " Neighbor " + str(j+1) + " j_id: " + str(journey_id) + " dtw_dist: " + str(dtw_distance))
#         gmap.draw("TT " + str(i+1) + " Neighbor " + str(j+1) + ".html")

# 10-fold
import random

num_lines = sum(1 for l in open('../train_set.csv'))
size = int(num_lines / 10)
skip_idx = random.sample(range(1, num_lines), num_lines - size)

trainSet = pd.read_csv(
	'../train_set.csv', # replace with the correct path
    skiprows=skip_idx,
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
trajectories = trainSet["Trajectory"]
journeyPatternIds = trainSet['journeyPatternId']

samples = []
for i in range(0, len(trajectories.keys())):
    index = trajectories.keys()[i]
    samples.append([])
    for t, lon, lat in trajectories[index]:
        samples[i].append([lat, lon])

labels = []
for i in range(0, len(journeyPatternIds.keys())):
    index = journeyPatternIds.keys()[i]
    labels.append(journeyPatternIds[index])

kf = KFold(n_splits=10)
# neigh = KNeighborsClassifier(n_neighbors=5)
# clf = GridSearchCV(neigh, {}, cv=kf, n_jobs=-1)
# clf.fit(samples, labels)

sum_acc = 0
for train_index, test_index in kf.split(samples):
    # print(train_index)
    # print(test_index)
    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()
    for index in train_index:
        X_train.append(samples[index])
        y_train.append(labels[index])
    for index in test_index:
        X_test.append(samples[index])
        y_test.append(labels[index])
    # X_train, X_test = samples[train_index], samples[test_index]
    # y_train, y_test = labels[train_index], labels[test_index]

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    result = neigh.predict(X_test, haversine)

    acc = accuracy_score(y_test, result)
    sum_acc += acc
    print (acc)
    print (classification_report(y_test, result))
print("Average Accuracy: ", sum_acc / 10)