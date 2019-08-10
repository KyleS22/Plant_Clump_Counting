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
from tqdm import tqdm

from multiprocessing import Pool

def run_system_test(real_counts_csv, path_to_clump_images, path_to_model, out_path,
        save_file_name="system_test_results.csv", model_type="CNN", path_to_weights=None):
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
    
    model = utils.load_model(path_to_model, model_type, path_to_weights=path_to_weights)

    y_true, y_pred, per_row_results = _evaluate_model(model, real_counts, sorted_clumps)

    try:
        y_pred = [x[0] for x in y_pred]
    except:
        y_pred = y_pred

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    

    test_scores = utils.create_test_scores_dict(y_true, y_pred)

    utils.save_test_results(test_scores, out_path, file_name=save_file_name)

    conf_mat = metrics.conf_matrix(y_true, y_pred)
    classes = np.sort(np.unique(np.concatenate((np.rint(y_pred), np.rint(y_true)), 0))).astype(int)#sorted([int(x) for x in os.listdir(validation_data_dir)])
    

    df = pd.DataFrame(conf_mat, index = classes, columns=classes)


    df.to_csv(os.path.join(out_path, os.path.splitext(save_file_name)[0] + "_conf_matrix.csv"))
    
    _save_per_row_results(per_row_results, out_path)

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


def _save_per_row_results(per_row_results, out_dir):
    
    save_dir = os.path.join(out_dir, "per_row_results")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for row in per_row_results.keys():
        stitches = per_row_results[row]

        stitch_results = [['stitch_name', 'predicted_count', 'true_count', 'percent_error', 'absolute_error']]

        for stitch in stitches.keys():
            
            true, pred = stitches[stitch]

            percent_error = (abs(pred - true)/true) * 100
            absolute_error = abs(pred - true)

            try:
                result = [stitch, pred, true,  percent_error[0], absolute_error[0]]
            except:
                result = [stitch, pred, true, percent_error, absolute_error]

            stitch_results.append(result)
        
        stitch_results = np.asarray(stitch_results)
       

        df = pd.DataFrame(data=stitch_results)#stitch_results[1:, 1:], index=stitch_results[1:, 0], columns=stitch_results[0, 1:])
        

        df.to_csv(os.path.join(save_dir, row + ".csv"))
        

def _load_and_sort_clump_images(path_to_clump_images):
    """
    Get the file names of the clump images.  The filenames of theses images should follow the following pattern:
    `xx_yyyyyyyyy_zzz.png`. `xx` refers to the row number, `yyyyyyyyy` refers to the "images I used for stitching", and
    `zzz` refers to the clump number detected on the stitched image. Don't add them based on the first two characters
    
    :param path_to_clump_images: The path to the folder containing the clump images
    :returns: A dictionary of the following structure:
        {'xx' : 
            { 'yyyyyyyyy': ['full/path/to/clump_1.png', 'full/path/to/clump_2.png', ... ],
              'yyyyyyyyy': ...
            },
         'xx' :
            ...
            }
    :raises OSError: If the given file path does not exist
    """

    if not os.path.exists(path_to_clump_images):
        raise OSError("The given image path does not exist: {}".format(path_to_clump_images))
    
    images = os.listdir(path_to_clump_images)
    
    sorted_images = {}

    for image in images:
        image_split = image.rsplit('_')
        row_number = image_split[0]
        stitch_name = image_split[1]
        clump_number = image_split[2]
        
        if not row_number in sorted_images.keys():
            sorted_images[row_number] = {}

        if not stitch_name in sorted_images[row_number].keys():
            sorted_images[row_number][stitch_name] = []

        sorted_images[row_number][stitch_name].append(os.path.join(path_to_clump_images, image))

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

    per_row_results = {}
    pbar = tqdm(total = len(sorted_images.keys()))
    pbar.set_description("Evaluating Rows")

    for row in sorted_images.keys():
     
        stitches = sorted_images[row]
        
        stitch_results = {}

        for stitch in stitches.keys():
            
            images = stitches[stitch]
            row_counts = 0

            for image in images:
                
                images_to_predict = model.prepare_input_from_file(image)
                prediction = model.predict(images_to_predict) 

                row_counts += prediction
      
            try:
                true_count = int(real_counts.loc[real_counts[0] == row].values[0][1])
            except:
                continue
        
            true_counts.append(true_count)
            predicted_counts.append(row_counts)
            
            stitch_results[stitch] = (true_count, row_counts)

        per_row_results[row] = stitch_results
        
        pbar.update(1)

    return true_counts, predicted_counts, per_row_results



