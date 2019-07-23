"""
File Name: system_test.py

Authors: Kyle Seidenthal

Date: 22-07-2019

Description: A module containing functions that allow testing of the counting system on the designated test set

"""
import sys
sys.path.append("../counting_CNN")


import os
import csv

import pandas as pd
import numpy as np

from counting_CNN.model import CountingModel as CCNN
from keras.preprocessing import image as keras_image

import test_utils.metrics as metrics
import test_utils.utils as utils


def run_system_test(real_counts_csv, path_to_clump_images, path_to_model, out_path, save_file_name="system_test_results.csv", model_type="CNN"):
    """
    Evaluate the given model as part of the counting system
    
    :param real_counts_csv: The path to the ground truth counts file
    :param path_to_clump_images: The path to the directory containing the clump images
    :param path_to_model: The path to the saved model to evaluate
    :param out_path: The path to save the test results in 
    :param save_file_name: The name of the file that will contain the results. Default is 'system_test_results.csv
    :param model_type:  The type of the model being evaluated.  Default is CNN 
    :returns: None
    """

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    real_counts = _read_real_counts_csv(real_counts_csv)
    sorted_clumps = _load_and_sort_clump_images(path_to_clump_images)
    
    model = utils.load_model(path_to_model, model_type)

    y_true, y_pred = _evaluate_model(model, real_counts, sorted_images)

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, save_file_name=save_file_name)

def _read_real_counts_csv(path_to_csv):
    """
    Read in the csv containing ground truth counts for each row in a field.  There should be two columns, the first
    being the filename of the row image containing the ground truth count, and the second should be the integer count of
    the number of plants in that row.
    
    :param path_to_csv: The path to the ground truth csv
    :returns: A pandas dataframe representation of the csv
    :raises OSError: If the given file does not exist
    """
    
    if not os.path.exists(path_to_csv):
        raise OSError("The given ground truth csv file does not exist: {}".format(path_to_csv))

    return pd.read_csv(path_to_csv, header=None)

def _load_and_sort_clump_images(path_to_clump_images):
    """
    Get the file names of the clump images.  The filenames of theses images should follow the following pattern:
    <filename>_<clump_number>, where clump number is simply an arbitrary identifier that distinguishes a clump image
    from other clump images belonging to the same row.
    
    :param path_to_clump_images: The path to the folder containing the clump images
    :returns: A dictionary of lists, where the keys are the filenames - corresponding to each row image - and the lists
              are the lists of images belonging to the same filename group
    :raises OSError: If the given file path does not exist
    """

    if not os.path.exists(path_to_clump_images):
        raise OSError("The given image path does not exist: {}".format(path_to_clump_images))
    
    images = os.listdir(path_to_clump_images)
    
    sorted_images = {}

    for image in images:
        image_split = image.rsplit('_', 1)
        filename = image_split[0]
        

        if sorted_images[filename] is None:
            sorted_images[filename] = []
        
        sorted_images[filename].append(os.path.join(path_to_clump_images, image))

    return sorted_images




def _evaluate_model(model, real_counts, sorted_images):
    """
    Evaluate the given model based on the real counts using the sorted images
    
    :param model: The model to evaluate
    :param real_counts: A pandas dataframe containing the real counts
    :param sorted_images: A dictionary of images that have been sorted by row
    :returns: The test results for the model (y_true, y_pred), lists of the corresponding values
    """

    true_counts = []
    predicted_counts = []

    for row in sorted_images.keys():
        images = sorted_images[row]

        row_counts = 0

        for image in images:
            
            images = model.prepare_input_from_file(image)
             
            prediction = model.predict(images) 

            row_counts += prediction
       
        true_count = real_counts.loc[real_counts[0] == row]
        
        true_counts.append(true_count)
        predicted_counts.append(row_counts)
        
    return true_counts, predicted_counts


  
