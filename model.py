
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#Read lines from csv file
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile) 
    for line in reader:
        lines.append(line)
        

#Split the lines for training and validating
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

correction = 0.2 # used to adjust the steering angle for left & right camera images
image_path = '../data/IMG/'

'''
Generator used to generate data from disk in batches instead of storing all images in memory
This method also augment the images - horizontal flipping 
''' 
def generator(samples, batch_size=120):
    num_samples = len(samples)
    
    #Generator processes lines in csv and generates 6 images per line
    #3 images - center, left and right camera images
    #all 3 images are flipped to generate additional 3 images
    batch_size = int(batch_size/6)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            #Read centre, left and right images
            #Steering angle is corrected for Left and right images
            #Correction required to treat them as if generated from centre camera
            images = []
            angles = []
                
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                center_file_name = (batch_sample[0]).split('/')[-1]
                left_file_name = (batch_sample[1]).split('/')[-1]
                right_file_name = (batch_sample[2]).split('/')[-1]

                center_image = cv2.imread(image_path + center_file_name)
                left_image = cv2.imread(image_path + left_file_name)
                right_image = cv2.imread(image_path + right_file_name)

                images.extend([center_image, left_image, right_image])
                angles.extend([steering_center, steering_left, steering_right])
                
            #Augment all images by flipping horizontally to produce more training data
            #Flipping is helpful to generate clock-wise & counter-clock wise turnings
            #so that the model trains better
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)

            #Convert to numpy array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            
            yield shuffle(X_train, y_train)

            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Cropping2D

#LeNet model - not used
#Initially tried with this model but switched to Nvidia
def LeNet(input_shape=(160, 320, 3)):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    '''
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))         
    model.add(Dense(84, activation='relu'))
    #model.add(Dropout(0.3)) 
    model.add(Dense(1, activation='linear'))
    '''
    model.add(Flatten())
    model.add(Dense(128))
    #model.add(Dropout(0.3))         
    model.add(Dense(84))
    #model.add(Dropout(0.3)) 
    model.add(Dense(1))

    return model

#Nvidia network model with 5 Convolution layers and four fully connected nuerons
def Nvidia(input_shape=(160, 320, 3), dropout=0.3):
    model = Sequential()

    #Data pre-processing
    #Normalizing the data and then mean centering using Lambda
    model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=input_shape))
    
    #Crop unwanted parts of image which would be unnecessary, distracting the model
    #Top 70 rows of pixels - sky, trees etc
    #Bottom 25 rows, where car's hood appears in images
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    
    #4 Fully connected layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(dropout))
    model.add(Dense(50))
    model.add(Dropout(dropout))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model

'''
Method to visualize the loss metrics i.e. Training loss & Validation loss
'''
def visualize_loss(history_object):
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    


# compile and train the model using the generator function

train_generator = generator(train_samples, batch_size=120)
validation_generator = generator(validation_samples, batch_size=120)

row, col, ch = 160, 320, 3  # Image size

input_shape = (row, col, ch)

model = Nvidia(input_shape)

#Compile model

model.compile(loss='mse', optimizer='adam')

#Note - for each line in csv file, 6 images are generated.
#3 - center, left & right camera images ( per line * 3 images)
#and each image is augmented by horizontal flip (3 images * 2 augmented)
history_object = model.fit_generator(train_generator, samples_per_epoch= \
len(train_samples)*3*2, validation_data=validation_generator, \
nb_val_samples=len(validation_samples)*3*2, nb_epoch=4, verbose=1)

#model.compile(loss = 'mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 5)

visualize_loss(history_object)

model.save('model.h5')