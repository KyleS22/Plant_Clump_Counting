"""
File Name: validation.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: A module containing functions that allow easy validataion of models

"""



import os
import csv

import pandas as pd
import numpy as np

from counting_CNN.model import CountingModel as CCNN
from keras.preprocessing import image as keras_image

import test_utils.metrics as metrics
import test_utils.utils as utils


def run_validation(validation_data_dir, path_to_model, out_path, save_file_name="validataion_test.csv", model_type="CNN", path_to_weights=None):
    """
    Run the validation set to get metrics on the given model
    
    :param validation_data_dir: The directory containing the validation images
    :param path_to_model: The path to the model save file
    :param out_path: The path to save the output results csv file
    :param save_file_name: The name of the output file.
                           Default is 'validation_test.csv'
    :param model_type: The model type
    :param path_to_weights: The path to the weights file for a CNN model.
                                 Default is None
    :returns: None
    """
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    image_paths, y_true = _get_validation_data(validation_data_dir)
    
    model = utils.load_model(path_to_model, model_type, path_to_weights)
    
    predictions = model.predict_generator(validation_data_dir, len(image_paths))
    
    try:
        y_pred = [x[0] for x in predictions]
    except:
        y_pred = predictions
    
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, file_name=save_file_name)

    conf_mat = metrics.conf_matrix(y_true, y_pred)
    classes = np.sort(np.unique(np.concatenate((np.rint(y_pred), np.rint(y_true)), 0))).astype(int)#sorted([int(x) for x in os.listdir(validation_data_dir)])
    

    df = pd.DataFrame(conf_mat, index = classes, columns=classes)

    df.to_csv(os.path.join(out_path, os.path.splitext(save_file_name)[0] + "_conf_matrix.csv"))


def _validate(model, image_paths, true_labels):
    """
    Run validation on the given model with the given data
    
    :param model: The model to validate
    :param image_paths: A list of the paths to the images to validate on
    :param true_labels: A list of the true labesl for each of the images in image_paths
    :returns: A list of true_labels and a list of the predicted values by the model
    """

    y_true = []
    y_pred = []

    images = []

    for image, label in zip(image_paths, true_labels):

        img = model.prepare_input_from_file(image)
        

        images.append(img)

        prediction = model.predict(img)

        y_true.append(label)
        y_pred.append(prediction)


    return y_true, y_pred
    

def _get_validation_data(validation_dir):
    """
    Return a list of the image paths and their true labels
    
    :param validation_dir: The directory to get the data from
    :returns: A list of image paths, and the true labels for each image
    """

    labels = os.listdir(validation_dir)

    image_paths = []
    true_labels = []

    for label in labels:
        images = os.listdir(os.path.join(validation_dir, label))

        for image in images:
            image_paths.append(os.path.join(validation_dir, label, image))
            true_labels.append(int(label))

    return image_paths, true_labels
            
    
