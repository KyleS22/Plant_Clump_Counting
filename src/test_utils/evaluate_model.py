"""
File Name: 

Authors: {% <AUTHOR> %}

Date: 12-08-2019

Description: {% <DESCRIPTION> %}

"""
import pandas as pd
import numpy as np
import os

import test_utils.metrics as metrics
import test_utils.utils as utils


def evaluate_model(path_to_results_dir, out_path):
    

    combined = _read_and_combine_per_row_results(path_to_results_dir)

    _compute_model_results(combined, out_path)

    _compute_per_row_results(path_to_results_dir, out_path)

    mean_counts = _read_and_combine_mode(path_to_results_dir)

    _compute_mode_results(mean_counts, out_path)

def _compute_per_row_results(data_dir, out_path):

    files = os.listdir(data_dir)

    for filename in files:
        df = _read_results_csv(os.path.join(data_dir, filename))

        df['absolute_error'] = abs(df['true_count'] - df['predicted_count'])
        
        df = df.sort_values(by='absolute_error')

        df.to_csv(os.path.join(out_path, filename))
    
    
def _compute_model_results(combined_predictions, out_path):
    
    y_pred = combined_predictions.as_matrix(columns=['predicted_count'])
    y_true = combined_predictions.as_matrix(columns=['true_count'])
    
    y_pred = [x[0] for x in y_pred]
    y_true = [x[0] for x in y_true]

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, file_name="model_results.csv")

    
def _read_and_combine_mode(data_dir):
    
    files = os.listdir(data_dir)
    
    data = []
    true = []

    for filename in files:
        df = _read_results_csv(os.path.join(data_dir, filename))
        
        row_num = int(filename.split(".")[0])

        df.insert(loc=0, column='row', value=row_num)
        
        mean_count = df['predicted_count'].mean()
        true_count = df['true_count'].mean()
        #df['absolute_error'] = abs(df['true_count'] - df['predicted_count'])
        #
        #df = df.sort_values(by='absolute_error')
        #
        #num_rows = len(df.index)

        #mode_row = int(num_rows/2)

        #data.append(df.iloc[[mode_row]])
        data.append(mean_count)
        true.append(true_count)

    #combined = pd.concat(data)

    return data, true




def _compute_mode_results(combined, out_path):
    

    #y_pred = combined.as_matrix(columns=['predicted_count'])
    #y_true = combined.as_matrix(columns=['true_count'])
    

    
    #y_pred = [x[0] for x in y_pred]
    #y_true = [x[0] for x in y_true]


    y_pred = combined[0]
    y_true = combined[1]

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, file_name="model_mode_results.csv")




def _read_results_csv(path_to_csv):
    return pd.read_csv(path_to_csv, header=1)

def _read_and_combine_per_row_results(data_dir):

    files = os.listdir(data_dir)
    data = []

    for filename in files:
        df = _read_results_csv(os.path.join(data_dir, filename))
        
        row_num = int(filename.split(".")[0])

        df.insert(loc=0, column='row', value=row_num)
        data.append(df)
        
    combined = pd.concat(data)

    return combined




