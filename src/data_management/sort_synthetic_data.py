"""
File Name: sort_synthetic_data.py

Authors: Kyle Seidenthal

Date: 14-07-2019

Description: A script to sort the synthetic images into their labels for use with the model training scripts

"""

import os
import argparse
from tqdm import tqdm

from shutil import copyfile

def sort(data_path, out_path):
    """
    Creates a new directory where the images in data_path are sorted into folders with their corresponding labels
    
    :param data_path: The path to get the images from
    :param out_path: The path to store the sorted data
    :returns: None
    """

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    images = os.listdir(data_path)

    pbar = tqdm(images)
    pbar.set_description("Sorting Synthetic Images")
     
    for image in images:
        parts = image.split('_')
        

        label = str(int(parts[0]))
        
        if 'bin.png' in parts:
            pbar.update(1)
            continue

        image_name = parts[1]
        
        label_dir = os.path.join(out_path, label)
        

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        image_path = os.path.join(label_dir, image_name)
        
        old_image_path = os.path.join(data_path, image)

        copyfile(old_image_path, image_path)
        
        pbar.update(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sort synthetic images into label folders")

    parser.add_argument("data_dir", help="The path to the images.")
    parser.add_argument("out_dir", help="The path to store the sorted images at.")

    args = parser.parse_args()

    sort(args.data_dir, args.out_dir)
