#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/ModelBC.png "Model Visualization"
[image2]: ./examples/graph.png "MSE Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* Track2RightLane.mp4 video of Track 2 driving on right lane. Trained with only Track 2 data.
* Track1CenterLane.mp4 video of Track 1 driving on center lane. Trained with data from both Tracks.
* Track2CenterLane.mp4 video of Track 2 driving on center lane. Trained with data from both Tracks.


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Modified the drive.py to set the speed to 20Mph.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with a combination of 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 64-78) 

The model includes RELU and Elu layers to introduce nonlinearity.

The data is normalized in the model using a Keras lambda layer (code line 65), and the data is preprocessed(Cropped) in the model using a Keras Cropping layer (code line 66). 

#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularization across all convolution and fully connected layers in order to reduce overfitting. 

The model was trained and validated using the data of both tracks to ensure that the model was not overfitting (code line 83-88). The model was tested by running it through the simulator on both tracks and ensuring that the vehicle could stay on the track over multiple laps.

Haven't tried dropout layers. Will do it as a follow up.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 4. Appropriate training data

I used the sample training data for Track 1 and collected training data for track 2 to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure it is overfitted first and then attempt to generalize it.

My first step was to start with the Lenet convolution neural network. I added normailization layer to it using keras lambda. 
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I trained with only center images of Track 1 data and ran the simulator to see how well the car was driving around. Unfortunately it was going off track in the turn after the bridge.
I then added flip augmentation but it didnt help.

I then switched to a convolution neural network model similar to the nVidia as suggested in the project guidelines.
I trained with both left and right images with a correction of 0.2. Trained it for 10 epochs and the car was able to successfully complete track 1.
But the car went off the track when I tested on Track 2. 

Next I wanted to make sure the car successfully completes track 2 when trained with only Track 2 data using the same model. For this I collected the data by driving on only the right lane of track 2.
On testing, the car came very close to hitting the barrier on few turns and it finally hit the barrier towards the end of the track. I then changed the correction to 0.3 for left and right images. 
After the correction change, the car was able succesfully complete track 2. 

Next I collected another set of track 2 data by driving in the center of the road as I felt the model might get confused when I combine center lane driving of Track 1 with right lane driving of Track 2.
Also to combat the overfitting, I modified the model to include L2 regularization.

Now with the combined data of both the tracks, flip augmentation, normalization and cropping as pre-processing, I was able to successfully complete the run on both the tracks using a single model.

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-79) consisted of a convolution neural network with the following layers and layer sizes.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

For Track 1 I used the sample data set provided. 

For Track 2, I first recorded a lap by driving on the right lane. 
Then I created another recording by driving in the center lane. 
I also recorded some recovery driving learn to move from right or left side to the center at multiple spots along the track.
To augment the data sat, I also flipped images and angles to generalize it and reduce overfitting.

My data set size after augmentation was 50368. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
I chose 10 as the ideal number of epochs. 

![alt text][image2]

#### 4. Next Steps
* Explore filtering 0 angle records.
* Try Dropout Layers
* Try Batch normailization  
