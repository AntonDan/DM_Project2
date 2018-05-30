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
import random

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

num_lines = sum(1 for l in open('../train_set.csv'))
size = int(num_lines / 40)
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

f = open('10fold_2p_out.txt', 'w')

sum_acc = 0
count = 1
for train_index, test_index in kf.split(samples):
    print("iteration ", count)
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
    print("iteration ", count, " created samples and labels")

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    result = neigh.predict(X_test, haversine)

    acc = accuracy_score(y_test, result)
    sum_acc += acc
    f.write("Accuracy" + str(acc))
    f.write(classification_report(y_test, result))
    count += 1
    print("iteration ", count, " finished.")
f.write("Average Accuracy: " + str(sum_acc / 10))