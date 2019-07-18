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
from keras.callbacks import Callback

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model, model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import numpy as np
import os
import shutil
import argparse
import csv
import sys
from datetime import datetime
import json

sys.path.append("../data_management")

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.mean_squared_error = []
        self.mean_absolute_error = []
        self.r_square = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mean_squared_error.append(logs.get('mean_squared_error'))
        self.mean_absolute_error.append(logs.get('mean_absolute_error'))
        self.r_square.append(logs.get('r_square'))

def r_square(y_true, y_pred):
    """
    Calculate the R squared value
    
    :param y_true: The true label
    :param y_pred: The predicted label
    :returns: The R Squared measure
    """
    
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - (SS_res/(SS_tot + K.epsilon()))

def _create_model():
    """
    Create the model for count estimation
    
    :returns: The model 
    """
    vgg16_model = VGG16(weights='imagenet', include_top=False)
        
    input_img = Input(shape=(224, 224, 3))  # adapt this if using `channels_first` image data format

    #x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)

    output_vgg16 = vgg16_model(input_img)
    vgg16_model.summary()
        
    x = Flatten(name='flatten')(output_vgg16)#(x)
    x = Dense(64, kernel_initializer='normal')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, kernel_initializer='normal')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, kernel_initializer='normal', name='Pre-Predictions')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, kernel_initializer='normal')(x)
    x = Dense(1, kernel_initializer='normal')(x)


    counter = Model(input_img, x)


    counter.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', r_square])
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


    os.mkdir(out_dir)
    

    model_json = model.to_json()

    json_name = name + ".json"
    weights_name = name + ".h5"
    summary_name = name + "_summary.txt"

    json_path = os.path.join(out_dir, json_name)
    weights_path = os.path.join(out_dir, weights_name)
    summary_path = os.path.join(out_dir, summary_name)

    with open(json_path, 'w') as json_file:
        json_file.write(model_json)
        
    model.save_weights(weights_path) 
    
    with open(summary_path, 'w') as summary_file:
        model.summary(print_fn=lambda x: summary_file.write(x + '\n'))

def _save_config_json(out_dir, args, start_time=None):
    """
    Save the configurations of this run to a JSON file
    
    :param out_dir: The directory to store the save file in
    :param args: The argparse arguments used to run the program
    :param start_time=None: The Start time of the program.  Default is none
    :returns: None
    """
    end_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    config = {}
    
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    config['start_time'] = start_time
    config['end_time'] = end_time
   
    out_path = os.path.join(out_dir, 'config.json')

    with open(out_path, 'w') as config_json:
        json.dump(config, config_json)


   
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
    
    loaded_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', r_square])
    loaded_model.summary()

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

def _create_test_generator(test_data_dir, batch_size=8, target_size=(224, 224)):
    """
    Create an image generator for the testing set
    
    :param test_data_dir: The directory containing the images
    :param batch_size: The batch size.  Defaults to 8
    :param target_size: The target image size for the generator.  Default is (224, 224)
    :returns: A testing image generator
    """

    test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')

    test_generator = test_datagen.flow_from_directory(test_data_dir, shuffle=False, target_size=target_size, batch_size=batch_size, class_mode='sparse')

    return test_generator
    

def _regression_flow_from_directory(flow_from_directory_gen, list_of_values):
    
    for x, y in flow_from_directory_gen:
        values = [list_of_values[int(y[i])] for i in range(len(y))]
        yield x, values

def _save_history(out_path, history):
    """
    Save the given Loss History object to csv
    
    :param out_path: The path to save the csv to
    :param history: The history object of the model to save
    :returns: None
    """
    
    
    with open(os.path.join(out_path, 'loss_history.csv'), 'w') as csvfile:
        fieldnames = ['loss', 'mean_squared_error', 'mean_absolute_error', 'r_square']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for loss, mse, mae, r_square in zip(history.losses, history.mean_squared_error, history.mean_absolute_error,
            history.r_square):

            writer.writerow({"loss": loss, "mean_squared_error": mse, "mean_absolute_error": mae, "r_square": r_square})

def _save_test_scores(out_path, test_scores):
    """
    Save the given test scores to a csv file
    
    :param out_path: The path to store the csv
    :param test_scores: A dictionary containing the test scores to save
    :returns: None
    """
    
    with open(os.path.join(out_path, 'test_scores.csv'), 'w') as csvfile:
        fieldnames = test_scores.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        writer.writerow(test_scores)

def _train_model(model, training_data_dir, test_data_dir, batch_size, num_epochs):
    """
    Train the given model.
    
    :param model: The model to train
    :param training_data_dir: The path to the training data
    :param test_data_dir: The path to the testing data
    :param batch_size: The batch size to use when training
    :param num_epochs: The number of epochs to use when training
    :returns: The trained model, its training history, and a dictionary contaning test scores
    """
    train_generator, validation_generator = _create_train_and_validation_generators(training_data_dir, batch_size)

    label_map = train_generator.class_indices.keys()
   
    list_of_values = [int(x) for x in label_map]
    
    reg_train_generator = _regression_flow_from_directory(train_generator, list_of_values)
    reg_validation_generator = _regression_flow_from_directory(validation_generator, list_of_values)
   
    history = LossHistory()

    model.fit_generator(reg_train_generator, steps_per_epoch=train_generator.samples // batch_size,
    validation_data=reg_validation_generator, validation_steps = validation_generator.samples // batch_size, epochs =
    num_epochs, callbacks=[history])
    
    test_labels = os.listdir(test_data_dir)
    test_labels = [int(x) for x in test_labels]

    test_generator = _create_test_generator(test_data_dir)
    reg_test_generator = _regression_flow_from_directory(test_generator, test_labels)


    results = model.evaluate_generator(reg_test_generator, steps=test_generator.samples // test_generator.batch_size)
    
    print("TEST RESULTS")
    test_scores = {}
    for name, score in zip(model.metrics_names, results):
        test_scores[name] = score
        
        print("{}: {}".format(name, score))
         
    return model, history, test_scores
    
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

    trained_model, history, test_scores = _train_model(model, training_data_dir, test_data_dir, batch_size, num_epochs)

    if model_save_dir is not None:
        print("\nSaving model....")
        _save_model(trained_model, model_save_dir, name=model_name)
        _save_history(model_save_dir, history)
        _save_test_scores(model_save_dir, test_scores)
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

    trained_model, history, test_scores = _train_model(model, training_data_dir, test_data_dir, batch_size, num_epochs)
    
    if model_save_dir is not None:
        print("Saving model...")
        _save_model(trained_model, model_save_dir, name=model_name)
        _save_history(model_save_dir, history)
        _save_test_scores(model_save_dir, test_scores)
        print("Saved!\n\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a counting model.") 

    parser.add_argument('--model_save_dir', default=None, help="The directory to save model files to.  The model will not be saved if not specified.")
    parser.add_argument('--model_load_dir', default=None, help="The directory to load a model from.")
    parser.add_argument('training_data_dir', help='The directory to get training data from.')
    parser.add_argument('test_data_dir', help='The directory to get testing data from.')
    parser.add_argument('--batch_size', default=8, type=int, help="The batch size to use when training.")
    parser.add_argument('--num_epochs', default=100, type=int, help="The number of epochs to use when training.")
    parser.add_argument('--model_name', default="model", help="The name of the model.")

    args = parser.parse_args()
    
    start_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    
    if args.model_save_dir is not None and os.path.exists(args.model_save_dir):
        print("The model save path you have chosen already exists.  Please choose another.")
        sys.exit(1)

    if args.model_load_dir is None:
        create_and_train_new_model(args.training_data_dir, args.test_data_dir, args.batch_size, args.num_epochs, model_save_dir=args.model_save_dir, model_name=args.model_name)

    else:
        load_and_retrain_model(args.model_load_dir, args.training_data_dir, args.test_data_dir, args.batch_size, args.num_epochs, model_name=args.model_name, model_save_dir=args.model_save_dir)

    _save_config_json(args.model_save_dir, args, start_time=start_time)
