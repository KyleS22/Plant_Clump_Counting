"""
File Name: 

Authors: Kyle Seidenthal

Date: 22-07-2019

Description: A CNN model for plant clump counting

"""

from keras.utils import Sequence
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, BatchNormalization
from keras.preprocessing import image as keras_image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model, model_from_json, load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import regularizers

import skimage.io as io
from skimage.transform import rescale
from skimage.util import pad

import numpy as np
import os
import shutil
import argparse
import csv
import sys
from datetime import datetime
import json
from tqdm import tqdm

from time import time
from keras.callbacks import TensorBoard
from keras import losses

import matplotlib.pyplot as plt

from keras import backend as K

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.mean_squared_error = []
        self.mean_absolute_error = []
        self.r_square = []
        self.count_acc = []
        self.mean_absolute_percentage_error = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mean_squared_error.append(logs.get('mean_squared_error'))
        self.mean_absolute_error.append(logs.get('mean_absolute_error'))
        self.r_square.append(logs.get('r_square'))
        self.count_acc.append(logs.get('count_accuracy'))
        self.mean_absolute_percentage_error.append(logs.get('mean_absolute_percentage_error'))

    def save(self, out_path):
        """
        Save the history to a csv
        
        :param out_path: The path to save the csv file
        :returns: None
        """

        with open(os.path.join(out_path, 'loss_history.csv'), 'w') as csvfile:
            fieldnames = ['loss', 'mean_squared_error', 'mean_absolute_error', 'r_square', 'count_acc',
            'mean_absolute_percentage_error']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            
            for loss, mse, mae, r_square, count_acc, mape in zip(self.losses, self.mean_squared_error, self.mean_absolute_error,
                self.r_square, self.count_acc, self.mean_absolute_percentage_error):

                writer.writerow({"loss": loss, "mean_squared_error": mse, "mean_absolute_error": mae, "r_square": r_square,
                "count_acc": count_acc, "mean_absolute_percentage_error": mape})


    
class EncoderCountingModel: 

    def __init__(self, save_dir="./TEMP_MODEL_OUT", use_checkpoint=True, name="model"):
        self.model = None
        self.name = name
        self.save_dir = save_dir
        self.checkpointer = None
        self.encoder_checkpointer = None
        self.encoder = None
        
        self.target_image_size = (112, 112)

        self._encoder_architecture()

        if use_checkpoint:
            self._init_model_checkpointer(save_dir)
            self._init_encoder_checkpointer(save_dir)

    def _init_model_checkpointer(self, out_path):
        """
        Initilizes a model checkpointer for the model
        
        :param out_path: The path to store model checkpoints in
        :returns: None
        """
       
        self._make_model_out_dir()

        save_name = os.path.join(out_path, self.name + ".h5")

        self.checkpointer = ModelCheckpoint(save_name, monitor='val_mean_absolute_percentage_error', mode='min', verbose=1,
                save_best_only=True)

    def _init_encoder_checkpointer(self, out_path):

        self._make_model_out_dir()

        save_name = os.path.join(out_path, self.name + "_encoder.h5")

        self.encoder_checkpointer = ModelCheckpoint(save_name, monitor='val_mean_squared_error', mode='min', verbose=1,
        save_best_only=True)

    def compile(self):
        """
        Compiles the model for training with the predetrmined metrics
        
        :returns: None.
        """
        
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        self.model.summary()

    def _compile_model(self):
        """
        Compiles the model for training with the predetrmined metrics
        
        :returns: None.
        """
        
    
        metrics = ['mse', 'mae', 'mean_absolute_percentage_error', self._get_count_accuracy_func()]

        
        self.model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=metrics)
        
        self.model.summary()

    
    def _get_encoder(self, input_img):
        
        x = Conv2D(112, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(input_img)
        x = BatchNormalization()(x)
        x = Conv2D(112, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2), padding='same')(x)
       
        x = Conv2D(56, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(56, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2), padding='same')(x)
       
        x = Conv2D(28, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(28, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2), padding='same')(x)
       
        x = Conv2D(14, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        encoder = MaxPooling2D((2, 2), padding='same')(x)
        
        return encoder

    def _get_decoder(self, encoder):
    
        x = Conv2D(14, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(encoder)
        x = BatchNormalization()(x)
        x = Conv2D(14, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
       
        x = Conv2D(28, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(28, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
       
        x = Conv2D(56, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(56, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
       
        x = Conv2D(112, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(112, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
       
        decoder = Conv2D(3, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)

        return decoder


    def _encoder_architecture(self):

        input_img = Input(shape=(112, 112, 3))
        
        encoder = self._get_encoder(input_img)
        decoder = self._get_decoder(encoder)


        self.model = Model(input_img, decoder)

 
    def _set_up_architecture(self):
        """
        Compile the model architecture
        
        :returns: None.  The model will be compiled into this object
        """
        
        autoencoder = self.model

        input_img = Input(shape=(112, 112, 3))

        x = Flatten(name='flatten')(self._get_encoder(input_img))
        x = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='Count')(x)

        self.model = Model(input_img, x)
        
        for l1, l2 in zip(self.model.layers[:8], autoencoder.layers[:8]):
            l1.set_weights(l2.get_weights())

        for layer in self.model.layers[:8]:
            layer.trainable = False
           
                  
    def prepare_input_from_file(self, file_path):
        """
        Loads and processes the given file for input to the model
        
        :param file_path: The path to the file
        :returns: The loaded and processed image, ready to use in the model
        """
   
        # load the image
        img = io.imread(file_path)
        
        if max(img.shape) > self.target_image_size[0]:
            # Get scaling factor 
            scaling_factor = self.target_image_size[0] / max(img.shape)

            # Rescale by scaling factor
            img = rescale(img, scaling_factor, multichannel=True)
        
        # pad shorter dimension to be 112
        pad_width_vertical = self.target_image_size[0] - img.shape[0]
        pad_width_horizontal = self.target_image_size[0] - img.shape[1]
        
        
        pad_top = int(np.floor(pad_width_vertical/2))
        pad_bottom = int(np.ceil(pad_width_vertical/2))
        pad_left =  int(np.floor(pad_width_horizontal/2))
        pad_right = int(np.ceil(pad_width_horizontal/2))

        padded = pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
        
    
        images = np.vstack([padded])
        return np.expand_dims(images, axis=0)

    def _save_model_json(self):
        """
        Save the model architecture to JSON
        
        :returns: None
        """
        

        self._make_model_out_dir()

        model_json = self.model.to_json()

        out_name = os.path.join(self.save_dir, self.name + ".json")

        with open(out_name, 'w') as json_file:
            json_file.write(model_json)

    def _save_model(self):
        """
        Save the model to the given output directory
        
        :param out_dir: The directory to store the model file in
        :returns: None
        """
        
        self._make_model_out_dir()

        out_name = os.path.join(self.save_dir, self.name + ".h5")
        
        self.model.save(out_name)
        
    def load_model_file(self, model_json_path, model_weights_path):
        """
        Load the model from the given directory
        
        :param model_dir: The directory to load the model from
        :returns: None.  The loaded model will be initialized in self.model
        """
        
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        model = model_from_json(loaded_model_json)
    
        self.model = model
        
        model.load_weights(model_weights_path)
               

    def _train_encoder(self, training_data_dir, batch_size, num_epochs, validation_data_dir=None):
        

        train_generator = self._create_encoder_generator(training_data_dir, batch_size)
        validation_generator = self._create_encoder_generator(validation_data_dir, 8)

       
        history = LossHistory()

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        
        es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1, patience=50,
        min_delta=5)
        
            
        if self.checkpointer is not None:
            
            self.model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator, validation_steps = validation_generator.samples // 8, epochs =
            num_epochs, callbacks=[history, tensorboard, es, self.encoder_checkpointer], verbose=1)
            

        else:
            self.model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator, validation_steps = validation_generator.samples // 8, epochs =
            num_epochs, callbacks=[history, tensorboard, es], verbose=1)
            self._save_model()
             
                      
        self._save_model_json()  
        return history



    def train(self, training_data_dir, batch_size, num_epochs, validation_data_dir=None):
        """
        Train the given model.
        
        :param training_data_dir: The path to the training data
        :param batch_size: The batch size to use when training
        :param num_epochs: The number of epochs to use when training
        :param validation_data_dir: The directory to get validation data from. Default is None.  If left as None, the
                                    training data will be split into a training and validation set.
        :returns: The training history
        """

        self._train_encoder(training_data_dir, batch_size, num_epochs, validation_data_dir=validation_data_dir)
        self._set_up_architecture()
        self._compile_model()

        if validation_data_dir is None:
            train_generator, validation_generator = self._create_train_and_validation_generators(training_data_dir, batch_size)

        else:
            train_generator = self._create_generator(training_data_dir, batch_size)
            validation_generator = self._create_generator(validation_data_dir, 8)

       
        history = LossHistory()

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        
        es = EarlyStopping(monitor='val_mean_absolute_percentage_error', mode='min', verbose=1, patience=50,
        min_delta=0.5)
        
            
        if self.checkpointer is not None:
            
            self.model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator, validation_steps = validation_generator.samples // 8, epochs =
            num_epochs, callbacks=[history, tensorboard, es, self.checkpointer], verbose=1)
            

        else:
            self.model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator, validation_steps = validation_generator.samples // 8, epochs =
            num_epochs, callbacks=[history, tensorboard, es], verbose=1)
            self._save_model()
             
                      
        self._save_model_json()  
        return history


    def predict(self, img):
        """
        Predict the count for the given image
        
        :img: An image to predict
        :returns: The predicted count for the image
        """
        return self.model.predict(img)[0][0]

    def predict_generator(self, data_dir, num_samples):
        """
        Predict the counts for all images in the given directory
        
        :param data_dir: The path to the image directory
        :param num_samples: The number of samples in the directory
        :returns: A list of the predictions for the sample images in the directory
        """
        

        val_gen = self._create_test_generator(data_dir)
        
        
        return self.model.predict_generator(val_gen, steps=num_samples/8)


    def evaluate(self, test_data_dir):
        """
        Evaluate the model on the testing data
        
        :param test_data_dir: The data to get test data from
        :returns: The results of the test
        """
        

        test_labels = os.listdir(test_data_dir)
        test_labels = [int(x) for x in test_labels]
        
         
        test_generator = self._create_test_generator(test_data_dir)
        
        results = self.model.evaluate_generator(test_generator, steps=test_generator.samples // test_generator.batch_size)
        
        test_scores = {}

        print("==============================")
        print("TEST RESULTS")
        
        for name, score in zip(self.model.metrics_names, results):
            test_scores[name] = score 

            print("{}: {}".format(name, score))

        return test_scores
  
    def _create_train_and_validation_generators(self, training_data_dir, batch_size, target_size=(112,112)):
        """
        Create an image generator for training and validation sets
        
        :param training_data_dir: The directory to get the training data from
        :param batch_size: The batch size to use
        :param target_size: The target image size for the generator.  Default is (224, 224)
        :returns: A tuple (train_generator, valid_generator)
        """
        

        raise NotImplementedException("Need to implement train/validation split and create two generators.")

        #return train_generator, validation_generator

    def _create_generator(self, data_dir, batch_size, target_size=(112, 112)):
        """
        Creates a data generator for the images
        
        :param data_dir: The directory to get the images from
        :param batch_size: The batch size to use
        :param target_size: The target size for the images.  Default is (224, 224)
        :returns: An image generator for the data
        """
        

        data_generator = ClumpImageGenerator(data_dir, batch_size=batch_size, target_image_size=target_size)

        return data_generator
        

    def _create_test_generator(self, test_data_dir, batch_size=8, target_size=(112, 112)):
        """
        Create an image generator for the testing set
        
        :param test_data_dir: The directory containing the images
        :param batch_size: The batch size.  Defaults to 8
        :param target_size: The target image size for the generator.  Default is (224, 224)
        :returns: A testing image generator
        """
        
        test_generator = ClumpImageGenerator(test_data_dir, batch_size, target_image_size=target_size)

        return test_generator
     
    
    def _create_encoder_generator(self, data_dir, batch_size, target_size=(112, 112)):
        
        data_generator = EncoderImageGenerator(data_dir, batch_size, target_image_size=target_size)

        return data_generator

    def _get_r_square_func(self):
        """
        Return the r_square function for the model
        
        :param self: Self
        :returns: The r_square function
        """
        
        def r_square(y_true, y_pred):
            """
            Calculate the R squared measure of the predicted values and their true values
            
            :param y_true: The true values, represented by a tensor
            :param y_pred: The predicted values, represented by a tensor
            :returns: The R squared value for the predicted values
            """
            from keras import backend as K

            SS_res =  K.sum(K.square(y_true - y_pred)) 

            SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
            
            return 1 - (SS_res/(SS_tot + K.epsilon()))
        
        return r_square 

    def _get_count_accuracy_func(self):
        """
        Return the count_accuracy() function
        
        :param self: Self
        :returns: The count_accuracy function
        """
        
        def count_accuracy(y_true, y_pred):
            """
            Calculate the count accuracy of the predicted values
            
            :param y_true: The true values
            :param y_pred: The predicted values
            :returns: The percentage of correct counts 
            """

            from keras import backend as K

            return K.mean(K.equal(y_true, K.round(y_pred)))

        return count_accuracy

    def _make_model_out_dir(self):
        """
        Creates the output directory to store the model
        
        :returns: None
        """
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


      



class ClumpImageGenerator(Sequence):
    """
    A Generator to load images into the counting model from a directory
    """
    
    def __init__(self, image_directory, batch_size, target_image_size=(112,112)):
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.image_filenames = []
        self.labels = []

        self._get_files_from_dir(image_directory)

    def _get_files_from_dir(self, directory):
        
        labels = os.listdir(directory)

    
        for label in labels:

            label_dir = os.path.join(directory, label)

            images = os.listdir(label_dir)

            for image in images:
                image_path = os.path.join(label_dir, image)
                self.image_filenames.append(image_path)
                self.labels.append(label)

        self.samples = len(self.image_filenames)
        print("FOUND: {} images belonging to {} classes.".format(self.samples, len(labels)))

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
        return np.array([self._read_and_process_image(file_name) for file_name in batch_x]), np.array(batch_y)
   
    def _read_and_process_image(self, file_name):
        
        # load the image
        img = io.imread(file_name)
        
        
        if img.shape[2] > 3:
            img = img[:, :, :3]
         
        if max(img.shape) > self.target_image_size[0]:
            # Get scaling factor 
            scaling_factor = self.target_image_size[0] / max(img.shape)

            # Rescale by scaling factor
            img = rescale(img, scaling_factor)
        
        # pad shorter dimension to be 112
        pad_width_vertical = self.target_image_size[0] - img.shape[0]
        pad_width_horizontal = self.target_image_size[0] - img.shape[1]
        
        
        pad_top = int(np.floor(pad_width_vertical/2))
        pad_bottom = int(np.ceil(pad_width_vertical/2))
        pad_left =  int(np.floor(pad_width_horizontal/2))
        pad_right = int(np.ceil(pad_width_horizontal/2))

        padded = pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
        
        # return the image
        return padded


class EncoderImageGenerator(Sequence):
    """
    A Generator to load images into the counting model from a directory
    """
    
    def __init__(self, image_directory, batch_size, target_image_size=(112,112)):
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.image_filenames = []
        self.labels = []

        self._get_files_from_dir(image_directory)

    def _get_files_from_dir(self, directory):
        
        labels = os.listdir(directory)

    
        for label in labels:

            label_dir = os.path.join(directory, label)

            images = os.listdir(label_dir)

            for image in images:
                image_path = os.path.join(label_dir, image)
                self.image_filenames.append(image_path)
                self.labels.append(label)

        self.samples = len(self.image_filenames)
        print("FOUND: {} images belonging to {} classes.".format(self.samples, len(labels)))

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        
        images = np.array([self._read_and_process_image(file_name) for file_name in batch_x])
    
        return images, images
   
    def _read_and_process_image(self, file_name):
        
        # load the image
        img = io.imread(file_name)
        
        
        if img.shape[2] > 3:
            img = img[:, :, :3]
         
        if max(img.shape) > self.target_image_size[0]:
            # Get scaling factor 
            scaling_factor = self.target_image_size[0] / max(img.shape)

            # Rescale by scaling factor
            img = rescale(img, scaling_factor)
        
        # pad shorter dimension to be 112
        pad_width_vertical = self.target_image_size[0] - img.shape[0]
        pad_width_horizontal = self.target_image_size[0] - img.shape[1]
        
        
        pad_top = int(np.floor(pad_width_vertical/2))
        pad_bottom = int(np.ceil(pad_width_vertical/2))
        pad_left =  int(np.floor(pad_width_horizontal/2))
        pad_right = int(np.ceil(pad_width_horizontal/2))

        padded = pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
        
        # return the image
        return padded
    
