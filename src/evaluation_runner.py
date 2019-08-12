"""
File Name: evaluation_runner.py

Authors: {% <AUTHOR> %}

Date: 12-08-2019

Description: {% <DESCRIPTION> %}

"""


import argparse
import sys

from test_utils import evaluate_model as em

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a counting model.") 

    parser.add_argument('prediction_dir', default=None, help="The path to the saved per-row predictions folder")
    parser.add_argument('out_path', help="The path to store output_files in")    

    args = parser.parse_args()
    
    em.evaluate_model(args.prediction_dir, args.out_path)
    print("Done!")
