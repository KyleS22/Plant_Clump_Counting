"""
File Name: visualize_conf_matrix.py

Authors: Kyle Seidenthal

Date: 09-08-2019

Description: A script to visualize a given confusion matrix from csv

"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def load_conf_mat(file_path):
    """
    Load the given csv into a dataframe
    
    :param file_path: The path to the csv file containing the confusion matrix
    :returns: A pandas dataframe representing the loaded csv
    """
    print(file_path)
    dataframe = pd.read_csv(file_path, index_col=0)
    return dataframe


def create_vis(df, save_path=None, silent=False):
    """
    Create a heatmap visualization of the confusion matrix
    
    :param df: The dataframe to visualize
    :param save_path: The path to save the figure to.  Default is None
    :param silent: Do not show the plot when generated
    :returns: None
    """
    

    plt.figure()
    sn.heatmap(df, annot=True, cmap=sn.cm.rocket_r)
    plt.xlabel("Predicted Count")
    plt.ylabel("True Count")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)

    
    if not silent:
        plt.show()


def run(file_path, save_path, silent):

    df = load_conf_mat(file_path)

    create_vis(df, save_path, silent)

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Visualize a confusion matrix from a csv") 

    parser.add_argument('file_path', default=None, help="The path to the saved model file")
    
    parser.add_argument('--save', default=None, help="The path to the place to save the figure")
    parser.add_argument('--silent', action='store_true', help='Do not display the plot when generated')
    args = parser.parse_args()

    run(args.file_path, args.save, args.silent)
    
