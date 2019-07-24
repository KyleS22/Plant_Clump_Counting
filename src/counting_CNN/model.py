"""
File Name: 

Authors: Kyle Seidenthal

Date: 22-07-2019

Description: A CNN model for plant clump counting

"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.preprocessing import image as keras_image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model, model_from_json, load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import regularizers

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

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mean_squared_error.append(logs.get('mean_squared_error'))
        self.mean_absolute_error.append(logs.get('mean_absolute_error'))
        self.r_square.append(logs.get('r_square'))
        self.count_acc.append(logs.get('count_accuracy'))

    def save(self, out_path):
        """
        Save the history to a csv
        
        :param out_path: The path to save the csv file
        :returns: None
        """

        with open(os.path.join(out_path, 'loss_history.csv'), 'w') as csvfile:
            fieldnames = ['loss', 'mean_squared_error', 'mean_absolute_error', 'r_square', 'count_acc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            
            for loss, mse, mae, r_square, count_acc in zip(self.losses, self.mean_squared_error, self.mean_absolute_error,
                self.r_square, self.count_acc):

                writer.writerow({"loss": loss, "mean_squared_error": mse, "mean_absolute_error": mae, "r_square": r_square,
                "count_acc": count_acc})


    
class CountingModel: 

    def __init__(self, save_dir="./TEMP_MODEL_OUT", use_checkpoint=True, name="model"):
        self.model = None
        self.name = name
        self.save_dir = save_dir
        self.checkpointer = None
        
        self._set_up_architecture()

        if use_checkpoint:
            self._init_model_checkpointer(save_dir)
    

    def _init_model_checkpointer(self, out_path):
        """
        Initilizes a model checkpointer for the model
        
        :param out_path: The path to store model checkpoints in
        :returns: None
        """
       
        self._make_model_out_dir()

        save_name = os.path.join(out_path, self.name + ".h5")

        self.checkpointer = ModelCheckpoint(save_name, monitor='val_mean_squared_error', mode='min', verbose=1,
                save_best_only=True)
    def compile(self):
    
        metrics = ['mse', 'mae', self._get_r_square_func(), self._get_count_accuracy_func()]

        
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)
        
        self.model.summary()

        
    def _set_up_architecture(self):
        """
        Compile the model architecture
        
        :returns: None.  The model will be compiled into this object
        """
        vgg16_model = VGG16(weights='imagenet', include_top=False)
        
        input_img = Input(shape=(224, 224, 3))  # adapt this if using `channels_first` image data format

        #output_vgg16 = vgg16_model(input_img)
        #vgg16_model.summary()

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x= MaxPooling2D((2, 2), padding='same')(x)
        
        x = Flatten(name='flatten')(x)#(output_vgg16)#(x)
        x = Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.1))(x)
        x = Dropout(0.3)(x)
        x = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.1), name='Count')(x)


        self.model = Model(input_img, x)

            
    def prepare_input_from_file(self, file_path):
        """
        Loads and processes the given file for input to the model
        
        :param file_path: The path to the file
        :returns: The loaded and processed image, ready to use in the model
        """
        img = keras_image.load_img(file_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        #x = x.reshape(1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        #x = np.expand_dims(x, axis=0)
            
        images = np.vstack([x])
        
        return images

    def _save_model_json(self):

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
       # model.load_weights(model_weights_path)
        #model = load_model(model_dir, custom_objects={"r_square": self._get_r_square_func(), "count_accuracy":
        #    self._get_count_accuracy_func()})
    
        self.model = model
        #self.compile()
        model.load_weights(model_weights_path)
               
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

        if validation_data_dir is None:
            train_generator, validation_generator = self._create_train_and_validation_generators(training_data_dir, batch_size)

        else:
            train_generator = self._create_generator(training_data_dir, batch_size)
            validation_generator = self._create_generator(validation_data_dir, batch_size)

        label_map = train_generator.class_indices.keys()
       
        list_of_values = [int(x) for x in label_map]
        
        reg_train_generator = self._regression_flow_from_directory(train_generator, list_of_values)
        reg_validation_generator = self._regression_flow_from_directory(validation_generator, list_of_values)
       
        history = LossHistory()

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.5)
        
            
        if self.checkpointer is not None:
            self.model.fit_generator(reg_train_generator, steps_per_epoch=train_generator.samples // batch_size,
            validation_data=reg_validation_generator, validation_steps = validation_generator.samples // batch_size, epochs =
            num_epochs, callbacks=[history, tensorboard, es, self.checkpointer], verbose=1)
            
            validation_generator = self._create_generator(validation_data_dir, batch_size)
            reg_validation_generator = self._regression_flow_from_directory(validation_generator, list_of_values)

        else:
            self.model.fit_generator(reg_train_generator, steps_per_epoch=train_generator.samples // batch_size,
            validation_data=reg_validation_generator, validation_steps = validation_generator.samples // batch_size, epochs =
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
        
        label_map = val_gen.class_indices.keys()
       
        list_of_values = [int(x) for x in label_map]
    
        regression_gen = self._regression_flow_from_directory(val_gen, list_of_values)
        
        return self.model.predict_generator(regression_gen, steps=num_samples/8)


    def evaluate(self, test_data_dir):
        """
        Evaluate the model on the testing data
        
        :param test_data_dir: The data to get test data from
        :returns: The results of the test
        """
        

        test_labels = os.listdir(test_data_dir)
        test_labels = [int(x) for x in test_labels]
        
         
        test_generator = self._create_test_generator(test_data_dir)
        reg_test_generator = self._regression_flow_from_directory(test_generator, test_labels)
        
        results = self.model.evaluate_generator(reg_test_generator, steps=test_generator.samples // test_generator.batch_size)
        
        test_scores = {}

        print("==============================")
        print("TEST RESULTS")
        
        for name, score in zip(self.model.metrics_names, results):
            test_scores[name] = score 

            print("{}: {}".format(name, score))

        return test_scores
  
    def _create_train_and_validation_generators(self, training_data_dir, batch_size, target_size=(224,224)):
        """
        Create an image generator for training and validation sets
        
        :param training_data_dir: The directory to get the training data from
        :param batch_size: The batch size to use
        :param target_size: The target image size for the generator.  Default is (224, 224)
        :returns: A tuple (train_generator, valid_generator)
        """
        
        # Create generators for the data
        train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last', zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(training_data_dir, target_size=target_size,
            batch_size=batch_size, subset="training", class_mode='sparse')
        
        validation_generator = train_datagen.flow_from_directory(training_data_dir, target_size=target_size,
        batch_size=batch_size, subset="validation", class_mode='sparse')
        
        return train_generator, validation_generator

    def _create_generator(self, data_dir, batch_size, target_size=(224, 224)):
        """
        Creates a data generator for the images
        
        :param data_dir: The directory to get the images from
        :param batch_size: The batch size to use
        :param target_size: The target size for the images.  Default is (224, 224)
        :returns: An image generator for the data
        """
        
        datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last', zoom_range=0.2, horizontal_flip=True)

        data_generator = datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size,
                class_mode='sparse')

        return data_generator
        

    def _create_test_generator(self, test_data_dir, batch_size=8, target_size=(224, 224)):
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
     
    
    def _regression_flow_from_directory(self, flow_from_directory_gen, list_of_values):
        """
        A function that allows using generators for regression values
        
        :param flow_from_directory_gen: An image generator
        :param list_of_values: The list of possible values for the true labels
        :returns: A generator that represents the images and true labels for each
        """
            
        for x, y in flow_from_directory_gen:
            values = [list_of_values[int(y[i])] for i in range(len(y))]
            yield x, values

    

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


      





        
