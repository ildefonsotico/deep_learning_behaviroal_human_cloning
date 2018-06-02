import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.misc

def resize(image, new_dim):
    """
    Resize a given image according the the new dimension
    :param image:
        Source image
    :param new_dim:
        A tuple which represents the resize dimension
    :return:
        Resize image
    """
    return scipy.misc.imresize(image, new_dim)

def crop(image, top_percent, bottom_percent):
    """
    Crops an image according to the given parameters
    :param image: source image
    :param top_percent:
        The percentage of the original image will be cropped from the top of the image
    :param bottom_percent:
        The percentage of the original image will be cropped from the bottom of the image
    :return:
        The cropped image
    """
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]

def generator(samples, batch_size=2):
    num_samples = len(samples)

    while 1:# Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:


                current_path = 'data/IMG/'
                filename_center = batch_sample[0].split('/')[-1]
                filename_left = batch_sample[1].split('/')[-1]
                filename_right = batch_sample[2].split('/')[-1]

                img_center = cv2.imread(current_path + filename_center)
                img_left = cv2.imread(current_path + filename_left)
                img_right = cv2.imread(current_path + filename_right)

                if img_center is None:
                    print("Image Center path incorrect: ", img_center)
                    continue
                if img_left is None:
                    print("Image Left path incorrect: ", img_left)
                    continue
                if img_right is None:
                    print("Image Right path incorrect: ", img_right)
                    continue

                img_center = crop(img_center, 0.35, 0.1)
                img_left = crop(img_left, 0.35, 0.1)
                img_right = crop(img_right, 0.35, 0.1)
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2292
                steering_left = steering_center + correction
                steering_right = steering_center - correction


                # add images and angles to data set
                images.append(img_center)
                measurements.append(steering_center)

                images.append(img_left)
                measurements.append(steering_left)

                images.append(img_right)
                measurements.append(steering_right)

            augmentation_imgs, augmentation_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmentation_imgs.append(resize(image, (64,64)))
                augmentation_measurements.append(measurement)
                augmentation_imgs.append(resize(cv2.flip(image, 1), (64,64)))
                augmentation_measurements.append(measurement * -1.0)

            # trim image to only see section with road
            X_train = np.array(augmentation_imgs)
            Y_train = np.array(augmentation_measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)




samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.3)


train_generator = generator(train_samples, batch_size=2)
validation_generator = generator(validation_samples, batch_size=2)

print("X Train: ",train_generator)
print("Y Train: ",train_generator)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam

learning_rate = 1e-4

model = Sequential()
#model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64,64,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3,subsample=(1,1), border_mode='same', activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(1,1), border_mode='same', activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer=Adam(learning_rate))

model.fit_generator(train_generator, samples_per_epoch= (6*len(train_samples)), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
#model.fit(X_train, Y_train, validation_split=0.3, shuffle=True, nb_epoch=5)

model.save('model_nvidia_3cameras_cropping_generator_modified_end.h5')