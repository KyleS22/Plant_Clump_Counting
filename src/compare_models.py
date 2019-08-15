"""
File Name: compare_models.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: A script to compare the validation results of a set of models

"""

import os
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description="Compare a set of models") 

parser.add_argument('result_paths', nargs='+', help="The path to the saved results file.  Can be listed by using multiple times.")
parser.add_argument('out_path', help="The path to store the comparison csv file
        to.")
parser.add_argument('--is_dir', action="store_true", help="If the given path is a directory, get all files in the directory and compare.")

args = parser.parse_args()


if args.is_dir:
    test_files = os.listdir(args.result_paths[0])
    
    test_files = [os.path.join(args.result_paths[0], x) for x in test_files]

else:
    test_files = args.result_paths

summary = None

for test_file in test_files:
    
    data = pd.read_csv(test_file)

    model_name = os.path.split(test_file)[1]


    data.insert(loc=0, column='Model_Name', value=model_name)

    if summary is None:
        summary = data

    else:
        summary = pd.concat([summary, data], ignore_index=True, sort=False)


sorted_summary = summary.sort_values(by=['accuracy'], ascending=False)

sorted_summary.to_csv(args.out_path)
