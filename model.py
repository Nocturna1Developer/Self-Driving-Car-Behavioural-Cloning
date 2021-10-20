'''
## CREDIT ##

#Keras Testing solution, 
#Transfer Learning Lab,
#Generator sample code,

### Github References:

https://github.com/felipemartinezs/CarND-Behavioral-Cloning-P3/blob/main/model.py (used as reference),

https://github.com/lewisHome/p3_Behavioural-Cloning (used as reference),

### Generator Code:

https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/95d62426-4da9-49a6-9195-603e0f81d3f1/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a

### Other Code: (flipping/correction):

https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/95d62426-4da9-49a6-9195-603e0f81d3f1/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/2cd424ad-a661-4754-8421-aec8cb018005

https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/95d62426-4da9-49a6-9195-603e0f81d3f1/lessons/818a5b8e-44b3-42f9-9921-e0e0e49f104e/concepts/ca8f22f8-7d7d-4989-ba30-9e850dd42bf8
 
### Offcial Udacity Video
https://www.youtube.com/watch?v=rpxZ87YFg0M&ab_channel=Udacity

'''
import pickle
import numpy as np
import tensorflow as tf
import os
import numpy as np
import cv2
import sklearn
import csv
import math

from sklearn.preprocessing import LabelBinarizer
from urllib.request import urlretrieve
from zipfile import ZipFile
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Lambda
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

samples = []
with open ('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                # Center Data
                images.append(center_image)
                angles.append(center_angle)

                # Flip Data                
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
            # Trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Preprocess image here
model = Sequential()

# Crop image here
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape =(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Add Convolutions and dropouts here
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.09))

model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('relu'))
#model.add(Dropout(0.09))

model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('relu'))
#model.add(Dropout(0.09))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
#model.add(Dropout(0.09))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
#model.add(Dropout(0.05))

# Flatten Image here
model.add(Flatten())

model.add(Dense(100)) 
model.add(Dropout(0.15))
model.add(Activation('relu'))
model.add(Dense(75)) 
model.add(Dropout(0.15))
model.add(Dense(50)) 
model.add(Activation('relu'))
#model.add(Dense(25))
#model.add(Dropout(0.15))
model.add(Dense(10))
model.add(Dropout(0.15))
model.add(Activation('relu'))
model.add(Dense(1)) 
#model.add(Dropout(0.15))

# Comile model here
model.compile(loss='mse',optimizer='adam')

# Fit data here
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs = 7, verbose = 1)

# Save model here
model.save('model2.h5')

# Keras method to print the model summary
model.summary()