"""
File Name: fix_madhiyar_results.py

Authors: {% <AUTHOR> %}

Date: 13-08-2019

Description: {% <DESCRIPTION> %}

"""

import pandas as pd
import numpy as np
import os

OUT_PATH = "../system_testing_results/UNet/CapsNet/per_row_results/"
INPUTS = "~/Unet_output.csv"


def _load(path):
    df = pd.read_csv(path)
    df.drop(df.tail(2).index, inplace=True)
    return df

def _load_real_counts(path_to_csv):
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

def _split_all(preds, real_counts):

    per_row_results = {}

    for index, row in preds.iterrows():
        filename = row[0]
        prediction = row[1]

        split_filename = filename.split("_")

        row_num = split_filename[0]
        stitch_name = split_filename[1]
        true_count = int(real_counts.loc[real_counts[0] == str(int(row_num))].values[0][1])

        
        if row_num not in per_row_results.keys():
            per_row_results[row_num] = pd.DataFrame(data=[["stitch_name", "predicted_count", "true_count"],
                [stitch_name, prediction, true_count]])

        else:
            df = per_row_results[row_num]

            df2 = pd.DataFrame([[stitch_name, prediction, true_count]])

            df3 = pd.concat([df, df2])

            per_row_results[row_num] = df3
                        

           # per_row_results[row_num] = per_row_results[row_num].append([stitch_name, prediction, true_count])


    
    for row in per_row_results.keys():

        df = per_row_results[row]

        df.to_csv(OUT_PATH + row + ".csv")



if __name__ == "__main__":

    df = _load(INPUTS)
    real_counts = _load_real_counts("/home/kyle/Plant_Counting_Data/real_counts.csv")
    
    _split_all(df, real_counts)

