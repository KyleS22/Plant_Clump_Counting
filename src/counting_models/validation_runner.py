"""
File Name: validation_runner.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: A command line script to validate a model

"""

import argparse

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from test_utils import validation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validate a counting model.") 

    parser.add_argument('model_path', default=None, help="The path to the saved model file")
    parser.add_argument('validation_data_dir', help='The directory to get validation data from.')
    parser.add_argument('test_result_path', help="The directory to store test results in.")
    parser.add_argument('test_name', help="The name for the output test files.")
    parser.add_argument('model_type', help="The type of the model being used")
    
    parser.add_argument('--path_to_weights', default=None, help="The path to the saved weights for a CNN")

    args = parser.parse_args()
    

    if args.model_type == "CNN" and args.path_to_weights is None:
        print("Validation of a CNN model requires the use of the '--path_to_weights' parameter.")
        sys.exit(1)

    validation.run_validation(args.validation_data_dir, args.model_path, args.test_result_path,
            save_file_name=args.test_name, model_type=args.model_type, path_to_weights=args.path_to_weights)

    print("Done!")
