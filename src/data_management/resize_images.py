"""
File Name: resize_images.py

Authors: Kyle Seidenthal

Date: 07-08-2019

Description: A script to resize and pad the images to a standard size

"""
import skimage.io as io
from skimage.transform import rescale
from skimage.util import pad

import numpy as np
import os
import argparse
import sys


def read_and_process_image(file_name, target_image_size=(112, 112)):
    """
    Read the given image and resize it to the target size.  Uses padding to keep the aspect ratio
    
    :param file_name: The path to the image to resize
    :param target_image_size: A tuple representing the size the new image should be 
    :returns: The resized image
    """
    
    # load the image
    img = io.imread(file_name)
    
    
    if img.shape[2] > 3:
        img = img[:, :, :3]
     
    if max(img.shape) > target_image_size[0]:
        # Get scaling factor 
        scaling_factor = target_image_size[0] / max(img.shape)

        # Rescale by scaling factor
        img = rescale(img, scaling_factor)
    
    # pad shorter dimension to be 112
    pad_width_vertical = target_image_size[0] - img.shape[0]
    pad_width_horizontal = target_image_size[0] - img.shape[1]
    
    
    pad_top = int(np.floor(pad_width_vertical/2))
    pad_bottom = int(np.ceil(pad_width_vertical/2))
    pad_left =  int(np.floor(pad_width_horizontal/2))
    pad_right = int(np.ceil(pad_width_horizontal/2))

    padded = pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
    
    return padded


def resize_images(input_dir):
    """
    Resize the images in the given input directory and save them in a separate directory with the same name +"_resized"
    
    :param input_dir: The path to the directory of images to resize
    :returns: None.  The resized images will be stored in a directory in the same place as the input_dir
    """

    out_dir = input_dir + "_resized"

    
    classes = os.listdir(input_dir)
    


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for c in classes:
        path = os.path.join(input_dir, c)
        out_path = os.path.join(out_dir, c)

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        images = os.listdir(path)

        

        for image in images:
            resized_img = _read_and_process_image(os.path.join(path, image))
            
            base = os.path.splitext(image)[0]

            io.imsave(os.path.join(out_path, base + ".png"), resized_img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Resize images to a predefined size.") 

    parser.add_argument('input_dir', default=None, help="The path to the images to rezize.")

    args = parser.parse_args()
    
    resize_images(args.input_dir) 
