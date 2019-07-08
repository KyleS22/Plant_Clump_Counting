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
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import shutil

import sys
sys.path.append("../data_management")

import create_train_and_test_dataset as train_test_split

DATA_DIR = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/cropped_images"


TRAIN_DIR = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/train"
TEST_DIR = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/test"


# Get the pretrained VGG weights
vgg16 = VGG16(weights=None, include_top=True)

# 20 is currently an arbitrary number of classes, representing clumps with a number of plants from 1-20
x = Dense(8, activation='softmax', name='predictions')(vgg16.layers[-2].output)

# Create the model
model = Model(input=vgg16.input, output=x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

images, image_labels = train_test_split.create_train_and_test_split(DATA_DIR, TRAIN_DIR, TEST_DIR, overwrite_old_data=True)

# Load the training Data
BATCH_SIZE = 5

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=BATCH_SIZE, subset="training")
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=BATCH_SIZE, subset="validation")

# Train 
model.fit_generator(train_generator, steps_per_epoch = train_generator.samples // BATCH_SIZE, validation_data =
    validation_generator, validation_steps = validation_generator.samples // BATCH_SIZE, epochs = 30)

test_labels = os.listdir(TEST_DIR)

print(test_labels)

test_images = []
test_image_labels = []

for label in test_labels:
    for image in os.listdir(os.path.join(TEST_DIR, label)):
            test_images.append(image)
            test_image_labels.append(label)


for img, label in zip(test_images, test_image_labels):

    image = load_img(os.path.join(TEST_DIR, label, img), target_size=(224, 224))
    ## convert the image pixels to a numpy array
    image = img_to_array(image)
    ## reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    prediction = model.predict_classes(image)

    print(prediction, label)



