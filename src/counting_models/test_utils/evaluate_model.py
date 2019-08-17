"""
File Name: evaluate_model.py

Authors: Kyle Seidenthal

Date: 12-08-2019

Description: A module containing functionality for computing performance
metrics for a model given its per row results.

"""
import pandas as pd
import numpy as np
import os

import test_utils.metrics as metrics
import test_utils.utils as utils


def evaluate_model(path_to_results_dir, out_path):
    """
    Run evaluation on the per_row results in the path_to_results dir, and save
    the outputs as csv files in out_path
    
    :param path_to_results_dir: The path to the model's per_row_results
    :param out_path: The path to the directory to save the results files
    :returns: None.
    """
    

    combined = _read_and_combine_per_row_results(path_to_results_dir)
    

    _compute_model_results(combined, out_path)

    _compute_per_row_results(path_to_results_dir, out_path)

    mean_counts = _read_and_combine_mean(path_to_results_dir)

    _compute_mean_results(mean_counts, out_path)


def _compute_per_row_results(data_dir, out_path):
    """
    Compute the results per row
    
    :param data_dir: The directory to get the per row results from
    :param out_path: The path to store the new per row results metrics
    :returns: None
    """

    files = os.listdir(data_dir)

    for filename in files:
        df = _read_results_csv(os.path.join(data_dir, filename))

        df['absolute_error'] = abs(df['true_count'] - df['predicted_count'])
        
        df = df.sort_values(by='absolute_error')

        df.to_csv(os.path.join(out_path, filename))
    
    
def _compute_model_results(combined_predictions, out_path):
    """
    Compute the overall results of the model given the combined per row results
    
    :param combined_predictions: A dataframe containing the per row results
                                 concatenated together.
    :param out_path: The path to the directory to store the model results file
    :returns: None
    """
    
    y_pred = combined_predictions.as_matrix(columns=['predicted_count'])
    y_true = combined_predictions.as_matrix(columns=['true_count'])
    
    
    try:
        y_pred = [x[0] for x in y_pred]
        y_true = [x[0] for x in y_true]
    except:
        pass

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, file_name="model_results.csv")

    
def _read_and_combine_mean(data_dir):
    """
    Read in the per row results files and gets the mean count prediction for
    each row
    
    :param data_dir: The directory to get the per row results from
    :returns: A list of the means of the predictions per row, and a list of the
              corresponding true counts
    """
    
    files = os.listdir(data_dir)
    
    data = []
    true = []

    for filename in files:
       
        df = _read_results_csv(os.path.join(data_dir, filename))
        

        row_num = int(filename.split(".")[0])

        df.insert(loc=0, column='row', value=row_num)
        
        mean_count = df['predicted_count'].mean()
        true_count = df['true_count'].mean()


        data.append(mean_count)
        true.append(true_count)


    return data, true




def _compute_mean_results(combined, out_path):
    """
    Compute the results of the model using the means of the rows 
    
    :param combined: A tuple containing a list of the means of the per row
                     predictions, and a list of the true counts
    :param out_path: The path to the directory to store the results in
    :returns: None
    """
    

    y_pred = combined[0]
    y_true = combined[1]

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, file_name="model_mean_results.csv")




def _read_results_csv(path_to_csv):
    """
    Read the given csv file into a dataframe
    
    :param path_to_csv: The path to the per row results file
    :returns: A dataframe containing the per row results
    """
    return pd.read_csv(path_to_csv)


def _read_and_combine_per_row_results(data_dir):
    """
    Read in the per row results in the given directory and combine them into
    one dataframe
    
    :param data_dir: The directory containing the per row results files
    :returns: A dataframe containing all results
    """

    files = os.listdir(data_dir)
    data = []

    for filename in files:
      
        df = _read_results_csv(os.path.join(data_dir, filename))
        row_num = int(filename.split(".")[0])

        df.insert(loc=0, column='row', value=row_num)
        data.append(df)
        
    combined = pd.concat(data)

    return combined




