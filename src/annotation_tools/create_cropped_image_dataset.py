"""
File Name: create_cropped_image_dataset.py

Authors: Kyle Seidenthal

Date: 04-07-2019

Description: A command line script to process a set of annotated images and crop them based on their bounding boxes.
             Stores the cropped images in a new directory

"""

import argparse
import os
import create_n_crop_bounding_box as cnc
import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm

def main(input_dirs, out_dir):
    """
    Process the images in the given directory and crop them 
    
    :param input_dirs: A list of directories that contain the images to be cropped
    :param out_dir: The directory to store the cropped images in
    :returns: None
    """
    

    for image_dir in input_dirs:
        
        images = os.listdir(image_dir)
       
        pbar = tqdm(images)
        pbar.set_description("Cropping %s" % image_dir)

        for image in images:
            
            try:
                cropped_images, image_labels, filename = cnc.cropped_bounding_boxes(image_dir,image)
            except:
                continue
                pbar.update(1)

            for cropped_image, label in zip(cropped_images, image_labels):

                cropped_out_dir = os.path.join(out_dir, str(label))

                if not os.path.exists(cropped_out_dir):
                    os.mkdir(cropped_out_dir)

                image_name = os.path.join(cropped_out_dir, filename)
                
                num = 1
                while os.path.exists(image_name):
                    name_split = os.path.splitext(image_name)
                    
                    image_name = name_split[0] + "_" + str(num) + name_split[1]
                    num += 1
                
                io.imsave(image_name, cropped_image)

            pbar.update(1) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Crop a given set of images using annotation files and output to a directory")

    parser.add_argument('-i', '--input_dirs', action="append", help="A list of directory names that will contain the\
     images to process.  Assumes that the annotations for the images are in a folder with the same name ffollowed by\
     _annotations.  Ex) If the images are in 'Images', then the annotations are in 'Images_annotations'.")

    parser.add_argument("output_dir", help="The directory to store the cropped images.")

    args = parser.parse_args()

    main(args.input_dirs, args.output_dir)
