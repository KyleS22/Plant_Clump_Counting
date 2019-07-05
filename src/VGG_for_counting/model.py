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

from sklearn.model_selection import train_test_split

import numpy as np
import os
import shutil

## load the model
#model = VGG16()
## load an image from file
#image = load_img('IPhone_Annotations/cropped_images/10/IMG_4444.JPG', target_size=(224, 224))
## convert the image pixels to a numpy array
#image = img_to_array(image)
## reshape data for the model
#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
## prepare the image for the VGG model
#image = preprocess_input(image)
## predict the probability across all output classes
#yhat = model.predict(image)
## convert the probabilities to class labels
#label = decode_predictions(yhat)
## retrieve the most likely result, e.g. highest probability
#label = label[0][0]
## print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))


# Get the pretrained VGG weights
vgg16 = VGG16(weights='imagenet', include_top=True)

# Add some fully connected layers
#x = Flatten(name='flatten')(vgg16.output)
#x = Dense(4096, activation='relu', name='fc1')(x)
#x = Dense(4096, activation='relu', name='fc2')(x)

# 20 is currently an arbitrary number of classes, representing clumps with a number of plants from 1-20
x = Dense(20, activation='softmax', name='predictions')(vgg16.layers[-2].output)

# Create the model
model = Model(input=vgg16.input, output=x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Split data into training and testing folders

data_dir = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/cropped_images"
labels = os.listdir(data_dir)


train_dir = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/train"
test_dir = "/student/kts135/School/Plant_Counting_Data/IPhone_Annotations/test"

try:
    os.mkdir(train_dir)
    os.mkdir(test_dir)

except:
    raise Exception("Could not make train and test dirs")

# This is the set of images to ignore, so we can see if the model can generalize to a label it has not seen
label_to_ignore = "6"

images = []
image_labels = []

for label in labels:
    
    print(label)
    if label == label_to_ignore:
        for img in os.listdir(os.path.join(data_dir, label)):
            shutil.copy(os.path.join(data_dir,label, img), os.path.join(test_dir, label))

    else:
        for image in os.listdir(os.path.join(data_dir, label)):
            images.append(image)
            image_labels.append(label)

os.mkdir(os.path.join(train_dir, label_to_ignore))

for i in range(20):
    if not str(i) in labels:
        os.mkdir(os.path.join(train_dir, str(i)))
        os.mkdir(os.path.join(test_dir, str(i)))

X_train, X_test, y_train, y_test = train_test_split(images, image_labels, test_size=0.2, random_state=42)

for image, label in zip(X_train, y_train):
    new_path = os.path.join(train_dir, label)
    old_path = os.path.join(data_dir, label, image)


    if not os.path.exists(new_path):
        os.mkdir(new_path)

    shutil.copy(old_path, new_path)

for image, label in zip(X_test, y_test):
    new_path = os.path.join(test_dir, label)
    old_path = os.path.join(data_dir, label, image)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    shutil.copy(old_path, new_path)



# Load the training Data
BATCH_SIZE = 20

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=BATCH_SIZE, subset="training")
validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=BATCH_SIZE, subset="validation")

# Train 
#model.fit(X_train, y_train, validation_data=(X_val, y_val))
model.fit_generator(train_generator, steps_per_epoch = train_generator.samples // BATCH_SIZE, validation_data =
    validation_generator, validation_steps = validation_generator.samples // BATCH_SIZE, epochs = 15)


# Test it
#predictions = model.predict(X_test)

# TODO: Compare to y_test


