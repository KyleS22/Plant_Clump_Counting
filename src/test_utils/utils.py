"""
File Name: utils.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: A module containing useful functions for evaluating models

"""

import os
import csv

from counting_CNN.model import CountingModel as CCNN
from test_utils import metrics

def load_model(path_to_model, model_type):
    """
    Load the given model for testing.
    
    :param path_to_model: The path to the saved model to test
    :param model_type: The type of model ex) CNN
    :returns: The loaded model
    :raise Exception: If the model type has not been accounted for here
    """

    if model_type.upper() == "CNN":
        model = CCNN()
        model.load_model_file(path_to_model)
    
    #TODO: Add else branch to load different types of models
       
    else:
        raise Exception("The model_type you have chosen is not currently supported.")
       
    
     
    return model

def save_test_results(test_results, out_path, file_name="system_test_scores.csv"):
    """
    Save the test scores dictionary to a CSV file
    
    :param test_results: The test result dictionary
    :param out_path: The path to save the test scores to
    :param file_name: The file name for the test scores.  Default is system_test_scores.csv
    :returns: None
    """
    

    with open(os.path.join(out_path, file_name), 'w') as csvfile:
        fieldnames = test_results.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        writer.writerow(test_results)

def create_test_scores_dict(y_true, y_pred):
    """
    Creates a dictionary containing all test metrics
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: A dictionary containing an entry for each metric
    """
   
    test_scores = {}

    test_scores["accuracy"] = metrics.accuracy(y_true, y_pred)
    test_scores["mean_squared_error"] = metrics.mean_squared_error(y_true, y_pred)
    test_scores["mean_absolute_error"] = metrics.mean_absolute_error(y_true, y_pred) 
    test_scores["r_square"] = metrics.r_square(y_true, y_pred)
        
    return test_scores

 
