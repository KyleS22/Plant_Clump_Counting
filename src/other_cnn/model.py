"""
File Name: 

Authors: Kyle Seidenthal

Date: 03-07-2019

Description: Training the VGG architecture to count plants in clumps

"""	
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
#from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import os
import shutil

import sys
sys.path.append("../data_management")

import create_train_and_test_dataset as train_test_split

DATA_DIR = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/cropped_images"


TRAIN_DIR = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/train"
TEST_DIR = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/test"

BATCH_SIZE = 8


# Create the model

input_img = Input(shape=(224, 224, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

#x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#autoencoder = Model(input_img, decoded)

#c = Flatten(name='flatten')(encoded)
#c = Dense(4096, activation='relu', name='fc1')(c)
#c = Dense(4096, activation='relu', name='fc2')(c)

# 20 is currently an arbitrary number of classes, representing clumps with a number of plants from 1-20
c = Flatten(name='flatten')(encoded)
c = Dense(8, activation='softmax', name='predictions')(c)



counter = Model(input_img, c)

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#autoencoder.summary()

counter.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
counter.summary()

# Create generators for the data
#train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last', shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

#encoder_train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode="input", subset="training")
#encoder_validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=BATCH_SIZE, class_mode="input", subset="validation")

# Train 
#model
#autoencoder.fit_generator(encoder_train_generator, steps_per_epoch = encoder_train_generator.samples // BATCH_SIZE, validation_data =
 #   encoder_validation_generator, validation_steps = encoder_validation_generator.samples // BATCH_SIZE, epochs = 100)



# Create generators for the data
train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last', shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=BATCH_SIZE, subset="training")
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=BATCH_SIZE, subset="validation")

counter.fit_generator(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator, validation_steps = validation_generator.samples // BATCH_SIZE, epochs = 50)

label_map = sorted(train_generator.class_indices.keys())


# Test
test_labels = os.listdir(TEST_DIR)

print(test_labels)

test_images = []
test_image_labels = []

for label in test_labels:
    for image in os.listdir(os.path.join(TEST_DIR, label)):
            test_images.append(image)
            test_image_labels.append(label)

num_correct = 0
total = 0

for img, label in zip(test_images, test_image_labels):

    image = load_img(os.path.join(TEST_DIR, label, img), target_size=(224, 224))
    ## convert the image pixels to a numpy array
    image = img_to_array(image)
    ## reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    prediction_prob = counter.predict(image)
    prediction = label_map[prediction_prob.argmax(axis=-1)[0]]


    if prediction == label:
        num_correct += 1
    
    total += 1
total = float(total)
num_correct = float(num_correct)
print(num_correct, total)
print(num_correct/total)

print("Testing accuracy %.3f " % (num_correct/total))

