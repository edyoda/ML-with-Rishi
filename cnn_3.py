# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:12:34 2021

@author: RISHBANS
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 19:35:00 2021

@author: RISHBANS
"""

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

#Initialize CNN
model = Sequential()

#Step - 1 Convolution
model.add(Convolution2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))

#Step - 2 MaxPooling
model.add(MaxPooling2D(pool_size=(2,2)))

#Step - 3 Flattening
model.add(Flatten())

#Full Connection
model.add(Dense(units = 128, activation = 'relu'))

#Output
model.add(Dense(units = 3, activation = 'softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Image Augmentation
from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, 
                               zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale = 1./255)



train_data = train_gen.flow_from_directory('dataset_3/train', target_size = (64,64),
                                           batch_size = 64, class_mode = 'categorical')

test_data = test_gen.flow_from_directory('dataset_3/val', target_size = (64,64),
                                           batch_size = 64, class_mode = 'categorical')


# Fit CNN to training set and test it on test set

model.fit_generator(train_data, steps_per_epoch = 200,
                    epochs = 30, validation_data = test_data)










