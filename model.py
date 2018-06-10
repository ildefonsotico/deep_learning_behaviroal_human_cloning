import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.misc
import random

from scipy.stats import rand_prob

HEIGHT,WIDTH, CHANNELS = 64,64,3 # resize the image to be 64x64
NEW_SHAPE = (HEIGHT,WIDTH)
ORIGINAL_SHAPE = (WIDTH, HEIGHT, CHANNELS)

def shear(image, steering_angle, shear_range=200):
    """
    Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk

    :param image:
        Source image on which the shear operation will be applied

    :param steering_angle:
        The steering angle of the image

    :param shear_range:
        Random shear between [-shear_range, shear_range + 1] will be applied

    :return:
        The image generated by applying random shear on the source image
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

def gamma_aleatory(image):
    """
    gamma aleatory is used in order to pre process data and also be used as a data augmentation tecnique

    :param image:
        image

    :return:
        New image generated with the gamma correction to the source image
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def resize(image, new_dim):
    """
    Resize a image with the new image received
    :param image:
        Source image
    :param new_dim:
        A tuple which represents the resize dimension
    :return:
        Resize image
    """
    return cv2.resize(image, new_dim, cv2.INTER_AREA)

def random_flip(image, steering_angle, flipping_prob=0.5):
    """
    Based on the outcome of an coin flip, the image will be flipped.
    If flipping is applied, the steering angle will be negated.

    :param image: Source image

    :param steering_angle: Original steering angle

    :return: Both flipped image and new steering angle
    """
    head = rand_prob.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


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

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:# Loop forever so the generator never terminates
        shuffle(samples)# each interaction the samples is shuffled

        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset+batch_size]
            #shuffle(batch_samples)
            images = []
            measurements = []
            cont1 = 0
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
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.235
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                rnd_image = np.random.randint(0, 3)#It used to randomly choose different camera
                if(rnd_image == 0):
                    img_center, steering_center = random_flip(img_center, steering_center)
                    images.append(img_center)
                    measurements.append(steering_center)

                if (rnd_image == 1):
                    img_left, steering_left = random_flip(img_left, steering_left)
                    images.append(img_left)
                    measurements.append(steering_left)

                if (rnd_image == 2):
                    img_right, steering_right = random_flip(img_right, steering_right)
                    images.append(img_right)
                    measurements.append(steering_right)


                cont1 = cont1 + 1

            cont = 0
            augmentation_imgs, augmentation_measurements = [], []
            for image, measurement in zip(images, measurements):
                cont = cont + 1


                rst = rand_prob.rvs(0.9)
                if rst == 1:
                    image, measurement = shear(image, measurement)

                image = crop(image, 0.35, 0.15)#crop each image by 35% on top and 15% on botton
                augmentation_imgs.append(resize(augment_brightness_camera_images(image), NEW_SHAPE))#generate e new image with randomly brigthness
                augmentation_measurements.append(measurement)

                #each image get a gamma randomly
                image = gamma_aleatory(image)
                augmentation_imgs.append(resize(image, NEW_SHAPE))
                augmentation_measurements.append(measurement)


            # trim image to only see section with road
            X_train = np.array(augmentation_imgs)
            Y_train = np.array(augmentation_measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)




samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#slip the datase, getting 20% to be used as a validation set... it was used before
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.optimizers import Adam

epochs = 8
samples_per_epoch = 30048
validation_samples = 7400
learning_rate = 1e-4
activation_relu = 'relu'

model = Sequential()
#model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=ORIGINAL_SHAPE))
model.add(Convolution2D(24,5,5,subsample=(2,2),border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
#model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3,subsample=(1,1), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3,subsample=(1,1), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(1164,  activation='relu'))
model.add(Dense(100,  activation='relu'))
model.add(Dense(50,  activation='relu'))
model.add(Dense(10,  activation='relu'))
model.add(Dense(1))
model.summary()
model.compile(optimizer=Adam(learning_rate), loss="mse" )

odel.fit_generator(train_generator, samples_per_epoch= (samples_per_epoch), validation_data=validation_generator, nb_val_samples=validation_samples, nb_epoch=epochs)
#model.fit(X_train, Y_train, validation_split=0.3, shuffle=True, nb_epoch=5)

model.save('model_nvidia_3cameras_cropping_35_15_generator_modified_relu_randon_brithness_30k_same_64_64.h5')