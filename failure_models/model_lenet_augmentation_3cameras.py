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
    filename_center = source_path.split('/')[-1]
    source_path = line[1]
    filename_left = source_path.split('/')[-1]
    source_path = line[2]
    filename_right = source_path.split('/')[-1]

    current_path = 'data/IMG/'
    img_center = cv2.imread(current_path + filename_center)
    img_left = cv2.imread(current_path + filename_left)
    img_right = cv2.imread(current_path + filename_right)
    #print("Image Center: ", img_center)
    #print("Image Left: ", img_left)
    #print("Image Right: ", img_right)

    if img_center is None:
        print("Image Center path incorrect: ", img_center)
        continue
    if img_left is None:
        print("Image Left path incorrect: ", img_left)
        continue
    if img_right is None:
        print("Image Right path incorrect: ", img_right)
        continue
    #images.append(image)
    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras

    # add images and angles to data set
    images.append(img_center)
    measurements.append(steering_center)

    images.append(img_left)
    measurements.append(steering_left)

    images.append(img_right)
    measurements.append(steering_right)

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

model.save('model_lenet_augmentation_3cameras.h5')