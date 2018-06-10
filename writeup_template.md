# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model has Keras lambda layer in order to normalize the model dataset. It is used just divind each image by 127.5 and then subtracting a offset (1.0).

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 204-214).

It will be presented bellow the architecture of the network. 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 215). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 180-184).
The validation set was done by spliting training dataset by 30%. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned manually to be 0.0001 (model.py line 111).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The dataset had way to recovery from each side. It also was used to make some curves smoothly in order to provide better way to handle with the curves.  


### Model Architecture and Training Strategy
#### 1. Pre-Processing Dataset

The dataset was passed through a crop funtion in order to dropout some noise on the images. The noises are sky, threes, hills, car hood, etc. These information will just distract the network with unecessary informations. 
In order to normalize the Dataset, each image was normalized using a simple method that is juts to subtract 127.5 by each pixel and then add an offeset by 1.
All dataset were resized to be (66, 200, 3) according with the NVIDIA paper. The dataset also was modified from RGB model to be YUV so that to reach NVIDIA specifications.

#### 2. Data Augmentation
In order to improve the training process and its generalization was improved the dataset with augmentation. The dataset was used to do that. The soution presented uses different data augmentation tecniques.

It was used all cameras installed on the vehicle, the letf, center and right camera. As known, the sterring angle is just provided by the center camera. Each camera used was also added a derivate steering wheel. The steering wheel derived consist in a simplest idea. It was added a factor (0.235) for the left steering wheel, in other hand it was subtracted (0.235) for the righ steering wheel. Each of them was match by its correspondent images either by righ or left camera. 

![cameras_behaviroal_human_driving](https://user-images.githubusercontent.com/19958282/41000142-6777c46c-68e3-11e8-9352-12f1e5736843.png)

Each image got was cropped by 35% on top of the image and 15% from the bottom. It was done in order to prevent distraction of the network with non important data.
![cameras_behaviroal_human_driving_cropped](https://user-images.githubusercontent.com/19958282/41000266-d38d854c-68e3-11e8-9985-ee84bf9cea11.png)

All  images was shuttle each interaction of the net. It was done like that in order to prevent overfitting and also reducing the training error. It is the most important thing in order to get a real good net trained. 

By each interaction the images were chose randomically. It was did in order to get just one image/measurement from three cameras (left, Center, Right).

Each image got was flipped by the cv library. Its own measurement also was flipped to be consistent. Basing in a probability the image used either could be original one or the flipped. 

The another tecnique used to create the dataset was generate the gamma by each image of the dataset chose. 

![gamma_sample19](https://user-images.githubusercontent.com/19958282/41196208-9724b428-6c11-11e8-8223-3486263b2c7d.png)
![gamma_sample20](https://user-images.githubusercontent.com/19958282/41196209-975a7798-6c11-11e8-9906-e650054c158f.png)
![gamma_sample21](https://user-images.githubusercontent.com/19958282/41196210-9790f278-6c11-11e8-916c-92c3e60e836c.png)
![gamma_sample22](https://user-images.githubusercontent.com/19958282/41196211-97c6f8be-6c11-11e8-83d2-5ae9b6e29a6b.png)
![gamma_sample10](https://user-images.githubusercontent.com/19958282/41196212-980107fc-6c11-11e8-8e11-48c4fdd45089.png)
![gamma_sample11](https://user-images.githubusercontent.com/19958282/41196213-984962ea-6c11-11e8-871e-bd3cb4727774.png)
![gamma_sample12](https://user-images.githubusercontent.com/19958282/41196214-987e5630-6c11-11e8-8a6f-e4aee5ca271d.png)
![gamma_sample13](https://user-images.githubusercontent.com/19958282/41196215-98b3ee4e-6c11-11e8-9a68-0804903c9c1e.png)
![gamma_sample14](https://user-images.githubusercontent.com/19958282/41196216-98ea00b0-6c11-11e8-9ec2-86bdc04f81d5.png)
![gamma_sample15](https://user-images.githubusercontent.com/19958282/41196217-99204616-6c11-11e8-9b9f-d2530e398324.png)


The another tecnique used to increase the dataset was generate brightness randomically. It was used this tecnique to better understanding. It also generates better generalization for the network. 

![gamma_sample29](https://user-images.githubusercontent.com/19958282/41196248-4fa4ecb6-6c12-11e8-87b9-9b1b73e0fa70.png)
![gamma_sample32](https://user-images.githubusercontent.com/19958282/41196249-4fdab30a-6c12-11e8-9144-7db4ec5caa93.png)
![gamma_sample4](https://user-images.githubusercontent.com/19958282/41196250-500f908e-6c12-11e8-8d0c-d024ef2d72c9.png)
![gamma_sample7](https://user-images.githubusercontent.com/19958282/41196251-50808186-6c12-11e8-8c35-c983540688d2.png)
![gamma_sample17](https://user-images.githubusercontent.com/19958282/41196252-50c01a6c-6c12-11e8-9138-a7259a52ec8d.png)
![gamma_sample19](https://user-images.githubusercontent.com/19958282/41196253-514925aa-6c12-11e8-9beb-0cdb5ac671e9.png)

The last tecnique used was shear randomically. It provides better data and improve the generalization. 

![shear_sample23](https://user-images.githubusercontent.com/19958282/41196288-d8a70d14-6c12-11e8-9733-125241846323.png)
![shear_sample1](https://user-images.githubusercontent.com/19958282/41196289-d8dcef38-6c12-11e8-8e14-13ce7b9ec930.png)
![shear_sample8](https://user-images.githubusercontent.com/19958282/41196291-d912ca0e-6c12-11e8-8078-225ef27fda79.png)
![shear_sample9](https://user-images.githubusercontent.com/19958282/41196292-d949d0d0-6c12-11e8-90a4-11c60ae00769.png)
![shear_sample11](https://user-images.githubusercontent.com/19958282/41196293-d9800a1a-6c12-11e8-9ea5-46376ce3f017.png)
![shear_sample13](https://user-images.githubusercontent.com/19958282/41196294-d9b75466-6c12-11e8-8deb-1fbd01f39397.png)
![shear_sample17](https://user-images.githubusercontent.com/19958282/41196295-d9fd54ca-6c12-11e8-987a-58d2edabe540.png)
![shear_sample18](https://user-images.githubusercontent.com/19958282/41196296-da61e1ec-6c12-11e8-97a2-ee43c9ad9738.png)

#### 3 Generator
As known, the dataset with data augmentation becomes too big. This quantity of the data usually can cause memory issues. In order to prevent this was used the generator function. This function works in general as common funtions, but instead of it lost its state, it keeps it, then always when you request it again, it will go one step further and provide you the next batch of the images. 
It decreases the performance but avoid memory issues. 
#### 1. Solution Design Approach


The model tryed to use different layers in order to get better accuracy such traning mode as validation mode. 
The model introduce RELU layers to introduce nonlinearity (code line 237 and 256). The better architeture was to use NvidiaNet with some maxpolling. It allowed better accuracy.

It uses a Convulution with filter with 5x5 size following for a RELU activation funtion. 
Next it uses a MaxPooling layer folloowing by a Convolution layer again with 5x5 filter. 
Next it uses MaxPooling again, then a flatten layer. 

The last 3 steps of the net used 3 fully connected layer starting by 1164 passing through 100, 50 and finally 1. 

It tested a lot of different architectures since the VGGNet and the NVIDIANet. All of the architectures was not able to do a specific curve (There are some models testes into the failure models folder). All of them went out of the road or hit the hills.  

In order to gauge how well the model was working, I split my image and steering angle dataset into a training and validation set by 30048 for training and 7400 for validation

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, all dataset was shuttle by each interaction. 
Then I got better tranning loss and validation.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, mainly on the next curve after the bridge. The folder failure show a lot failures test made. It was done a lot testing using differents data agumentation and net architecture. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

It will be provided two video. The first one focus on local vehicle camera. It is used according udacity project specifications. The another video is a top view. It shows better how the vehicle droves on the track. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 237-256) consisted of a implementation of the NVIDIA Net according with this [NVIDIA](https://github.com/ildefonsotico/deep_learning_behaviroal_human_cloning/files/2087943/end-to-end-dl-using-px.pdf). 
It was used RELU activation funtion so that provide nolinearity into the network. 

Here is a visualization of the architecture.

![deeplearning_architecture](https://user-images.githubusercontent.com/19958282/40879200-ded61036-6672-11e8-80da-c9d39828f468.png)

