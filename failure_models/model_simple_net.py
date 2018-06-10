import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

def SimpleNet(X_train, Y_train)



    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, validation_split=0.3, shuffle=True, nb_epoch=7)

    model.save('model_simple_fullyconnected.h5')

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path=line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    if image is None:
        print("Image path incorrect: ", current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)
print("X Train: ",X_train)
print("Y Train: ",Y_train)

SimpleNet(X_train, Y_train)