import pandas as pd
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors
import numpy as np
from gmplot import gmplot

from knn import KNeighborsClassifier
from haversine import haversine

# Read train set from disk
trainSet = pd.read_csv(
	'../train_set.csv', # replace with the correct path
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)

trajectories = trainSet["Trajectory"]
journeyPatternIds = trainSet['journeyPatternId']

# Read test set from disk
testSet = pd.read_csv(
	'../test_set_a1.csv', # replace with the correct path
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

# Create a KNN classifier and fit with train set
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(samples, labels)

# Get K=5 neighbors for each item in test set
result = neigh.kneighbors(test, haversine, 5)

# Display results and print info
for i, answer in enumerate(result):
    gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)

    longitudes = list()
    latitudes = list()
    for lat, lon in test[i]:
        longitudes.append(lon)
        latitudes.append(lat)

    gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map
    gmap.draw("Test Trip " + str(i+1) + ".html")

    for j, (journey_id, dtw_distance, trajectory) in enumerate(answer):
        longitudes = list()
        latitudes = list()
        for lat, lon in trajectory:
            longitudes.append(lon)
            latitudes.append(lat)

        gmap.plot(latitudes, longitudes, "red", edge_width=2)		# Plot it on the map
        print("Test Trip " + str(i+1) + " Neighbor " + str(j+1) + " j_id: " + str(journey_id) + " dtw_dist: " + str(dtw_distance))
        gmap.draw("TT " + str(i+1) + " Neighbor " + str(j+1) + ".html")