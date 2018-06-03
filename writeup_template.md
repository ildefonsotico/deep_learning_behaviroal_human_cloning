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


#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

The model tryed to use different layers in order to get better accuracy such traning mode as validation mode. 
The model introduce RELU layers to introduce nonlinearity (code line 96 and 98). The base archicture that worked better is a very simple one. 

It uses a Convulution with filter with 5x5 size following for a RELU activation funtion. 
Next it uses a MaxPooling layer folloowing by a Convolution layer again with 5x% filter. 
Next it uses MaxPooling again, then a flatten layer. 

The last 3 steps of the net used 3 fully connected layer starting by 120 passing through 84 and finally 1. 

It tested a lot of different architectures since the VGGNet and the NVIDIANet. All of the architectures was not able to do a specific curve (The .h5 files from all architectures I have tried are into the trained folder). All of them went out of the road or hit the hills. This simple one was the best approach found in terms of capacity to complete de circuit without hit or slipery for the edges of the curves. 

In order to gauge how well the model was working, I split my image and steering angle dataset into a training and validation set. 

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that use dropout with keep_prob of the 0.5. 
Then I got better tranning loss and validation.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, mainly on the next curve after the bridge. I improved the driving behavior in these cases just collecting more data about that section of the circuit.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 204-222) consisted of a implementation of the NVIDIA Net according with this PDF. 
It was used ELU activation funtion so that provide nolinearity into the network. ELU was used because it has better derivatives proximely zero, but there is low difference between ELU and ReLU actually.


Here is a visualization of the architecture.

![deeplearning_architecture](https://user-images.githubusercontent.com/19958282/40879200-ded61036-6672-11e8-80da-c9d39828f468.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
