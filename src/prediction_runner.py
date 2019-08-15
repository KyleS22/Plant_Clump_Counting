"""
File Name: validation_runner.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: A command line script to get the predictions from a model and store them.

"""

import argparse
import sys

from test_utils import create_prediction_csv as cpc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validate a counting model.") 

    parser.add_argument('model_path', default=None, help="The path to the saved model file.")
    parser.add_argument('validation_data_dir', help='The directory to get validation data from.')
    parser.add_argument('test_result_path', help="The directory to store test results in.")
    parser.add_argument('model_type', help="The type of the model being used.")
    parser.add_argument('real_counts_path', default=None, help="The path to the real_counts csv.")    

    parser.add_argument('--path_to_weights', default=None, help="The path to the saved weights for a CNN.")

    args = parser.parse_args()
    

    if args.model_type == "CNN" and args.path_to_weights is None:
        print("CNN models require that a weights file is used.  Please use the '--path_to_weights' option.")

    cpc.create_prediction_csv(args.real_counts_path, args.validation_data_dir, args.model_path,
        args.test_result_path, model_type=args.model_type,
        path_to_weights=args.path_to_weights)

    print("Done!")
