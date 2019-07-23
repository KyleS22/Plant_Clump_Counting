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

args = parser.parse_args()



summary = None

for test_file in args.result_paths:

    data = pd.read_csv(os.path.join(model_dir, test_file))

    model_name = test_file


    data.insert(loc=0, column='Model_Name', value=model_name)

    if summary is None:
        summary = data

    else:
        summary = pd.concat([summary, data], ignore_index=True, sort=False)


sorted_summary = summary.sort_values(by=['accuracy'], ascending=True)

print(sorted_summary)
