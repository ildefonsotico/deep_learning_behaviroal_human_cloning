import csv
import cv2
import numpy as np

lines = []

with open('../data.driving_log.csv') as csvfile:
    reader = csvfile.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path=line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG' + filename
    image = cv2.read(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

from Keras.models import Sequential
from Keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.3, shuffle=True, nb_epochs=7)

model.save('model_simple_fullyconnected.h5')