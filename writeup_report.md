# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center.jpg "Center"
[image2]: ./left.jpg "Recovery Image"
[image3]: ./right.jpg "Recovery Image"
[image4]: ./bef_flip.jpg "Normal Image"
[image5]: ./after_flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 file, video recording of vehicle driving autonomously around the track
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 layers using 5x5 filter size and another 2 layers with 3x3 filter size and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Used 0.2 as a correction angle to include side camera images in data set. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in opposite direction in track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architecture was derived in iterations. First I tried with a simple model then LeNet and finally cameup with a model inspired by the architecture published by Nvidia team.

First, I created a data loading pipeline using generators so that I could train and test with more data without consuing huge amount of memory. 

I created a simple model, when tested noticed that the MSE is high on both training and validatation data sets. This an evidense of underfitting. The obvious solution is to add more convolutions.

I've graduallly increased number of layers and achieved low mean square error for training set but with little higher for validation set. To avoid this overfitting, I introduced dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially near the lake and on the bridge. To improve the driving behaviour in these cases, I collected revovery data when the car is driving from side of the road back to center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with 5 convolution layers and 4 fully connected layers.

Before the first convolution layer, I've added a Lambda layer to do image normalization. Performing normalization in the network model allows normalizing the input images when tested in simulator as well, no additional code required to preprocess them.

Five convolutional layers were added to extract the image features fully instead of lines or shapes. First 3 layers with 5x5 kernals of 2x2 strides. Next 2 layers are with 3x3 kernal size.

Following the convolution layers are the four fully connected layers with output sizes of 100, 50, 10 and 1 (output layer) respectively.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center][image1]

I have also recorded by driving vehicle in opposite direction of the track so that the model generalizes well.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on track if it gets off to the side of the road.

These images show what a recovery looks like from left and right side of the road. :

![alt text][image2]
![alt text][image3]


To augment the data sat, I also flipped images and angles thinking that this would help model to train on how to steer clock-wise and also anti-clockwise directions. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had about 120000 number of data points. I then preprocessed this data by croping top and bottom rows of pixels to avoid unimportant details such as sky, trees and the vehicle bannet that appears in bottom portion of image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs found to be 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
