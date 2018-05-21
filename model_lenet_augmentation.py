import csv
import cv2
import numpy as np

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
        continue
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmentation_imgs, augmentation_measurements = [], []
for image, measurement in zip(images, measurements):
    augmentation_imgs.append(image)
    augmentation_measurements.append(measurement)
    augmentation_imgs.append(cv2.flip(image,1))
    augmentation_measurements.append(measurement*-1.0)

X_train = np.array(augmentation_imgs)
Y_train = np.array(augmentation_measurements)
print("X Train: ",X_train)
print("Y Train: ",Y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.3, shuffle=True, nb_epoch=5)

model.save('model_lenet_w_augmentation.h5')