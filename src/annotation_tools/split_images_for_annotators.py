"""
File Name: split_images_for_annotators.py

Authors: Kyle Seidenthal

Date: 17-06-2019

Description: A script to quickly divide up a set of images between a number of annotators, allowing for overlap for
inter-rater studies.

"""
import argparse
import os
import numpy as np
import shutil
from tqdm import tqdm
import random

def split_images(input_dir, output_dir, num_raters, overlap, rater_names=None):
    """
    Split the images in input_dir into a set of directories equal to num_raters.  Each directory will contain a subset
    of the images in input_dir, such that each contains only a number of duplicates equal to the specified overlap.
    
    :param input_dir: The directory to get the images from
    :param output_dir: The directory to store the split images in
    :param num_raters: The number of raters to split the images between
    :param overlap: The number of images to overlap between the raters, used for an inter-rater study
    :param rater_names: A list of the names of the raters, used only for naming the directories.  Default names will be
        used if this parameter is not specified.
    :returns: None
    """
    
    # Get the image names from the directory
    image_names = os.listdir(input_dir)
    num_images = len(image_names)
    
    # The number of images per rater, accounting for the overlap 
    images_per_rater = int(((num_images - overlap) / num_raters) + overlap)
    
    # Choose a random sample of the images to be the overlap images
    overlap_images = np.random.choice(image_names, overlap, replace=False)

    remaining_images = [img for img in image_names if img not in overlap_images]

    if rater_names is None:
        rater_names = ["rater_" + str(x) for x in range(1 , num_raters + 1)]
    
    # Creater rater dirs
    for name in rater_names:
        
        try:
            os.mkdir(os.path.join(output_dir, name))
        except:
            print("Failed to make directory %s", name)

            exit(1)
    
    # Make a directory for the overlap images
    try:
        os.mkdir(os.path.join(output_dir, "overlap_images"))
    except:
        print("Failed to make overlap dir")

        exit(1)

    for img in overlap_images:
        for name in rater_names:
            shutil.copyfile(os.path.join(input_dir,img), os.path.join(output_dir, name, img))
        
        shutil.copyfile(os.path.join(input_dir, img), os.path.join(output_dir, "overlap_images", img))   
    
    # Randomize the remaining images
    random.shuffle(remaining_images)
    
    cur_rater = 0

    pbar = tqdm(total=len(remaining_images))
    while len(remaining_images) is not 0:

        # Choose an image from the list, remove it, and copy it to the next rater
        img = remaining_images.pop(0)
        
        name = rater_names[cur_rater]

        shutil.copyfile(os.path.join(input_dir, img), os.path.join(output_dir, name, img))

        cur_rater = (cur_rater + 1) % num_raters

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Split the images in a given directory into a set of directories for a group of annotators.')
    
    parser.add_argument('input_dir', help="The path to the images to be split.")
    parser.add_argument('output_dir', help="The path to a directory to store the directories of split images.")
    parser.add_argument('num_raters', type=int, help='The number of raters to divide the images between.')
    parser.add_argument('overlap', type=int, help='The number of images to use for inter-rater agreement.')
    parser.add_argument('-n', '--rater_names', action="append", help='A list of the names of the raters. Specify each\
        element of the list as -n <name> ')

    args = parser.parse_args()
    
    if args.rater_names:
        split_images(args.input_dir, args.output_dir, args.num_raters, args.overlap, rater_names=args.rater_names)
    else:
        split_images(args.input_dir, args.output_dir, args.num_raters, args.overlap)
