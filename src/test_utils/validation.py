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


def run_validation(validataion_data_dir, path_to_model, out_path, save_file_name="validataion_test.csv", model_type="CNN"):
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    image_paths, true_labels = _get_validation_data(validation_data_dir)

    model = utils.load_model(path_to_model)

    y_true, y_pred = _validate(model, image_paths, true_labels)

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, save_file_name=save_file_name)

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

    for image, label in zip(image_paths, true_labels):

        img = model.prepare_input_from_file(image)

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

    return image_paths, trie_labels
            
    
