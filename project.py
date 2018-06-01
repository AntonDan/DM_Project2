import pandas as pd

from gmplot import gmplot
from ast import literal_eval
from random import randint

colors = ['red', 'green', 'blue', 'black', 'yellow']

# Read train set from disk
trainSet = pd.read_csv(
	'../train_set.csv',
	converters={"Trajectory": literal_eval},
	index_col='tripId'
)

trajectories = trainSet["Trajectory"]
journeyPatternIds = trainSet['journeyPatternId']

visited = list()
for i, color in zip(range(5), colors):
	gmap = gmplot.GoogleMapPlotter(53.383015, -6.237581, 12)	# Start Map roughly in Dublin

	gen = randint(0, len(trajectories.keys()) - 1)				# Generate a random number in range of applicable keys
	index = trajectories.keys()[gen]							# Get a random applicable index on the dataset
	while (journeyPatternIds[index] in visited):				# Do a check if we have already visualized the journeyPatternID
		gen = randint(0, len(trajectories.keys()) - 1)			# Generate a random index again if that is the case
		index = trajectories.keys()[gen]
	visited.append(journeyPatternIds[index])					# Append the journeyPatternID for duplicate checking

	longitudes = list()											# Create longitude points list
	latitudes = list()											# Create latitude points list
	for t, lon, lat in trajectories[index]:						# Fill them with the desired trip info
		longitudes.append(lon)
		latitudes.append(lat)

	gmap.plot(latitudes, longitudes, color, edge_width=5)		# Plot it on the map

	gmap.draw(str(index) + "_" + color + "_" + ".html")			# Export map in html format