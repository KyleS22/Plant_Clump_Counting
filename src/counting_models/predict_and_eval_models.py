"""
File Name: predict_and_eval_models.py

Authors: {% <AUTHOR> %}

Date: 16-08-2019

Description: A command line program to evaluate the performance of different
pipeline combinations

"""

import argparse
import sys
import os
import json
import csv
import pandas as pd

from test_utils import create_prediction_csv as cpc
from test_utils import evaluate_model as em

def _read_json_config(path_to_config):
    """
    Read in the required parameters from the config json
    
    :param path_to_config: The path to the config json
    :returns: A dictionary with the configurations
    """
    with open(path_to_config) as json_file:
        data = json.load(json_file)

        return data


if __name__ == "__main__":
    
    config = _read_json_config("./pred_eval_config.json")

    real_counts_path = os.path.expanduser(config["real_counts_path"])
    
    pred_out_path = os.path.expanduser(config["prediction_out_path"])
    
    total_models = len(config["detection_outputs"]) * len(config["models"])
    cur_model = 0

    for det_out in config["detection_outputs"]:
        


        for model in config["models"]:
            
            cur_model += 1

            print("Checking model {} of {}".format(cur_model, total_models))

            model_path = os.path.expanduser(model["path"])
            model_weights = os.path.expanduser(model["weights"])
            model_type = model["type"]
            model_name = model["name"]
           
            
            if model_weights == "None":
                model_weights = None

            det_name = os.path.basename(os.path.normpath(det_out))
        

            if not os.path.exists(pred_out_path): 
                os.mkdir(pred_out_path)
            
            model_pred_out_path = os.path.join(pred_out_path, det_name)
            
            
            if not os.path.exists(model_pred_out_path): 
                os.mkdir(model_pred_out_path)
        
            model_pred_out_path = os.path.join(model_pred_out_path, model_name)

            if not os.path.exists(model_pred_out_path): 
                os.mkdir(model_pred_out_path)

            cpc.create_prediction_csv(real_counts_path,
                    os.path.expanduser(det_out),
                    model_path, model_pred_out_path, model_type=model_type, path_to_weights=model_weights)

            em.evaluate_model(os.path.join(model_pred_out_path,
                "per_row_results"), model_pred_out_path)



    total_results = None

    for det_out in os.listdir(pred_out_path):

        path = os.path.join(pred_out_path, det_out)
        
        if not os.path.isdir(path):
            continue
         
        for model in os.listdir(path):
            csv_path = os.path.join(path, model, "model_results.csv")
        

            data = pd.read_csv(csv_path)
            if 'abs_count_diff' in data.columns:
                del data['abs_count_diff']

            data.insert(0, 'counting_model', model)
            data.insert(0, 'detection_model', det_out)
            
            if total_results is None:
                total_results = data

            else:
                total_results = pd.concat([total_results, data])

    print(total_results)
    total_results = total_results.sort_values(by="mean_absolute_error")
    print(total_results)
    total_results.to_csv(config["total_results_out_path"])
            



