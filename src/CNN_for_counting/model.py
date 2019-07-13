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
#from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.metrics import mean_squared_error

import numpy as np
import os
import shutil
import argparse

import sys
sys.path.append("../data_management")

#import create_train_and_test_dataset as train_test_split


def _create_model():
    """
    Create the model for count estimation
    
    :returns: The model 
    """
    vgg16_model = VGG16(weights=None, include_top=False)
        
    input_img = Input(shape=(224, 224, 3))  # adapt this if using `channels_first` image data format

    #x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)

    output_vgg16 = vgg16_model(input_img)

    x = Flatten(name='flatten')(output_vgg16)#(x)
    x = Dense(8, activation='softmax', name='Pre-Predictions')(x)
    x = Dense(1, kernel_initializer='normal')(x)


    counter = Model(input_img, x)


    counter.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    counter.summary()

    return counter


def _save_model(model, out_dir,  name="model"):
    """
    Save the model to JSON and the weights to h5 format
    
    :param model: The model to save
    :param out_dir: The directory to store the model files in
    :param name: The name of the model to use when creating file names.  Default is 'model'
    :returns: None
    """

    if not os.path.exists(out_dir):
        os.path.mkdir(out_dir)

    model_json = model.to_json()

    json_name = name + ".json"
    weights_name = name + ".h5"
    
    json_path = os.path.join(out_dir, json_name)
    weights_path = os.path.join(out_dir, weights_name)

    with open(json_path, 'w') as json_file:
        json_file.write(model_json)
        
    model.save_weights(weights_path) 
    

def _load_model(json_path, weights_path):
    """
    Load a model from a json, and its weights from an h5 file
    
    :param json_path: The path to the json model
    :param weights_path: The path to the weights file
    :returns: The loaded model
    """
    try:
        json_file = open(json_path, 'r')
    except:
        print("Could not load model json")
        sys.exit(1)

    loaded_json_model = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_json_model)

    loaded_model.load_weights(weights_path)

    return loaded_model

def _create_train_and_validation_generators(training_data_dir, batch_size, target_size=(224,224)):
    """
    Create an image generator for training and validation sets
    
    :param training_data_dir: The directory to get the training data from
    :param batch_size: The batch size to use
    :param target_size: The target image size for the generator.  Default is (224, 224)
    :returns: A tuple (train_generator, valid_generator)
    """
    
    # Create generators for the data
    train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last', shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(training_data_dir, target_size=target_size,
        batch_size=batch_size, subset="training", class_mode='sparse')
    
    validation_generator = train_datagen.flow_from_directory(training_data_dir, target_size=target_size,
    batch_size=batch_size, subset="validation", class_mode='sparse')
    
    return train_generator, validation_generator

def _regression_flow_from_directory(flow_from_directory_gen, list_of_values):
    
    for x, y in flow_from_directory_gen:
        values = [list_of_values[int(y[i])] for i in range(len(y))]
        yield x, values

def _train_model(model, training_data_dir, test_data_dir, batch_size, num_epochs):
    """
    Train the given model.
    
    :param model: The model to train
    :param training_data_dir: The path to the training data
    :param test_data_dir: The path to the testing data
    :param batch_size: The batch size to use when training
    :param num_epochs: The number of epochs to use when training
    :returns: The trained model
    """
    train_generator, validation_generator = _create_train_and_validation_generators(training_data_dir, batch_size)

    # TODO: The generators don't work for regression, because the folders imply a class structure

    label_map = train_generator.class_indices.keys()
   
    list_of_values = [int(x) for x in label_map]
    
    reg_train_generator = _regression_flow_from_directory(train_generator, list_of_values)
    reg_validation_generator = _regression_flow_from_directory(validation_generator, list_of_values)


    model.fit_generator(reg_train_generator, steps_per_epoch=train_generator.samples // batch_size,
    validation_data=reg_validation_generator, validation_steps = validation_generator.samples // batch_size, epochs =
    num_epochs)

    test_labels = os.listdir(test_data_dir)


    test_images = []
    test_image_labels = []

    for label in test_labels:
        for image in os.listdir(os.path.join(test_data_dir, label)):
            test_images.append(image)
            test_image_labels.append(label)

    predictions = []
    correct_values = []
    for img, label in zip(test_images, test_image_labels):

        image = load_img(os.path.join(test_dir, label, img), target_size=(224, 224))
        ## convert the image pixels to a numpy array
        image = img_to_array(image)
        ## reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
        prediction = model.predict(image)
        predictions.append(prediction)
        correct_values.append(int(label))
        
        print(str(prediction), label)


    print("Testing MSE %.3f " % mean_squared_error(predictions, correct_values))

    
    return model
    
def create_and_train_new_model(training_data_dir, test_data_dir, batch_size, num_epochs, model_save_dir=None, model_name="model"):
    """
    Create a new model and train it with the given data.  The model will be saved as a JSON and the weights will be
    stored in h5 format in the specified directory
    
    :param model_save_dir: The directory to save the model JSON and weights file in
    :param training_data_dir: The directory containing the training data
    :param test_data_dir: The directory containing the testing data
    :param batch_size: The batch size to use when training
    :param num_epochs: The number of epochs to use when training
    :param model_name: The name of the model, for naming the model json and weights file.  Default is 'model' 
    :returns: None
    """
    model = _create_model()

    trained_model = _train_model(model, training_data_dir, test_data_dir, batch_size, num_epochs)

    if model_save_dir is not None:
        print("\nSaving model....")
        _save_model(trained_model, model_save_dir, name=model_name)
        print("Saved!\n\n")

def load_and_retrain_model(model_path, training_data_dir, test_data_dir, batch_size, num_epochs, model_name="model", model_save_dir=None):
    """
    Load a saved model and its weights and retrain it on the given training and testing data.
    
    :param model_path: The path to the model json and weights files
    :param training_data_dir: The directory to get the training data from
    :param test_data_dir: The directory to get the testing data from
    :param batch_size: The batch size to use while training
    :param num_epochs: The number of epochs to use while training
    :param model_name: The name of the model, used to find the json and h5 files in the model_path. Default is 'model' 
    :returns: None
    """

    if not os.path.exists(model_path):
        raise OSError("The given model path does not exist: %s" % model_path)

    json_path = os.path.join(model_path, model_name + ".json")
    weights_path = os.path.join(model_path, model_name + ".h5")
    
    model = _load_model(json_path, weights_path)

    trained_model = _train_model(model, training_data_dir, test_data_dir, batch_size, num_epochs)
    
    if model_save_dir is not None:
        print("Saving model...")
        _save_model(trained_model, model_save_dir, name=model_name)
        print("Saved!\n\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a counting model.") 

    parser.add_argument('--model_save_dir', default=None, help="The directory to save model files to.  The model will not be saved if not specified.")
    parser.add_argument('--model_load_dir', default=None, help="The directory to load a model from.")
    parser.add_argument('training_data_dir', help='The directory to get training data from.')
    parser.add_argument('test_data_dir', help='The directory to get testing data from.')
    parser.add_argument('--batch_size', default=8, help="The batch size to use when training.")
    parser.add_argument('--num_epochs', default=100, help="The number of epochs to use when training.")
    parser.add_argument('--model_name', default="model", help="The name of the model.")

    args = parser.parse_args()

    if args.model_load_dir is None:
        create_and_train_new_model(args.training_data_dir, args.test_data_dir, args.batch_size, args.num_epochs, model_save_dir=args.model_save_dir, model_name=args.model_name)

    else:
        load_and_retrain_model(args.model_load_dir, args.training_data_dir, args.test_data_dir, args.batch_size, args.num_epochs, model_name=args.model_name, model_save_dir=args.model_save_dir)
