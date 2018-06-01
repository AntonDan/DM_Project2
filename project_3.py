from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd
from ast import literal_eval
from random import randint
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from knn import KNeighborsClassifier
from haversine import haversine

import random

# Read a percentage of random lines as test file for the K-fold
num_lines = sum(1 for l in open('../train_set.csv'))
size = int(num_lines / 40)  # "/ 40" means 2% of the actual file, can be adjusted
skip_idx = random.sample(range(1, num_lines), num_lines - size)

trainSet = pd.read_csv(
	'../train_set.csv',
    skiprows=skip_idx,
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)
trajectories = trainSet["Trajectory"]
journeyPatternIds = trainSet['journeyPatternId']

testSet = pd.read_csv(
	'../test_set_a2.csv',
    sep='\t',
    converters={"Trajectory": literal_eval}
)
testSet = testSet["Trajectory"]

# Reform data to be easier for our process
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

test = []
for i in range(0, len(testSet.keys())):
    index = testSet.keys()[i]
    test.append([])
    for t, lon, lat in testSet[index]:
        test[i].append([lat, lon])

# Create the csv with the results of the KNN on the requested file
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(samples, labels)
result = neigh.predict(test, haversine)

dic = {
    "Test_Trip_ID" : range(len(test)),
    "Predicted_JourneyPatternID" : result
}
out_df = pd.DataFrame(dic, columns=['Test_Trip_ID', 'Predicted_JourneyPatternID'])
out_df.to_csv("testSet_JourneyPatternIDs.csv", sep='\t', index=False)

# Do a 10 fold for our KNN and save results in file
kf = KFold(n_splits=10)

f = open('10fold_2p_out.txt', 'w')

sum_acc = 0
count = 1
for train_index, test_index in kf.split(samples):
    print("iteration ", count)

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