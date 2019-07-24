"""
File Name: validation_runner.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: A command line script to validate a model

"""

import argparse
import sys

from test_utils import validation, system_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validate a counting model.") 

    parser.add_argument('model_path', default=None, help="The path to the saved model file")
    parser.add_argument('validation_data_dir', help='The directory to get validation data from.')
    parser.add_argument('test_result_path', help="The directory to store test results in.")
    parser.add_argument('test_name', help="The name for the output test files.")
    
    parser.add_argument('model_type', help="The type of the model being used")
    parser.add_argument('--sys_test', action='store_true')
    parser.add_argument('--real_counts_path', default=None, help="The path to the real_counts csv.")    
    parser.add_argument('--path_to_weights', default=None, help="The path to the saved weights for a CNN")

    args = parser.parse_args()
    
    if args.sys_test:
        if args.real_counts_path is None:
            print("real_counts.csv required for sytem test!")
            sys.exit(1)

        system_test.run_system_test(args.real_counts_path, args.validation_data_dir, args.model_path,
            args.test_result_path, save_file_name=args.test_name, model_type=args.model_type)
    else:

        validation.run_validation(args.validation_data_dir, args.model_path, args.test_result_path,
            save_file_name=args.test_name, model_type=args.model_type, path_to_weights=args.path_to_weights)

    print("Done!")
