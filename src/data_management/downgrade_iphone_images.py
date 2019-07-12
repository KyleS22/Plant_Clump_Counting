"""
File Name: downgrade_iphone_images.py

Authors: Kyle Seidenthal

Date: 09-07-2019

Description: A module containing functionality to match the high quality IPhone images with the quality of the GroPro
images

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse

from skimage import data
from skimage import io
from skimage import data, exposure, img_as_float
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import rescale

from shutil import copyfile
from tqdm import tqdm

REFERENCE_PATH = "./GrowPro_Annotations/cropped_images/9/G0356568.JPG"
IMAGE_PATH = "./IPhone_Annotations/cropped_images/5/IMG_4444.JPG"

def _match_cumulative_cdf(source, template):
    """
    Match the histogram of the source image to the histogram of the template imatge, and then return the modified image
    
    :param source: The image whose histogram requires matching
    :param template: The image whose histogram will be matched
    :returns: The matched image
    """
    
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array

    src_quantiles = np.cumsum(src_counts).astype(np.float64) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float64) / template.size


    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)



def match_histograms(image, reference, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel. Taken from Scikit-Image
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape
    image_dtype = image.dtype

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched

def _scale_image(source, target):
    """
    Rescale the source image to a similar size as the target image
    
    :param source: The image to rescale
    :param target: The image whose size should be matched
    :returns: The rescaled source image
    """
    
    scaling_factor = (float(target.shape[0])/float(source.shape[0]) + float(target.shape[1])/float(source.shape[1]))/2

    img = rescale(source, scaling_factor)

    return img   

def match_images(source_path, target_path):
    """
    Match the scale and histogram of the source image to the target image
    
    :param source_path: The path to the source image
    :param target_path: The path to the target image
    :returns: The matched image
    """
    
    target = io.imread(target_path)
    source = io.imread(source_path)

    # Convert to HSV space because RGB will mess up the colours
    source = rgb2hsv(source)
    target = rgb2hsv(target)

    matched = match_histograms(source, target, multichannel=True)

    return hsv2rgb(matched)

def create_matched_dataset(sources_dir, targets_dir, output_dir):
    """
    Match all images in the sources_dir to the images in targets_dir, and store all processed images and the target
    images in the output_dir.  Assumes the sources_dir and targets_dir directories are structured such that the images
    are in folders corresponding to their labels within the given directory.
    
    :param sources_dir: The directory containing the source images
    :param targets_dir: The directory containing the target images
    :param output_dir: The directory to put the new dataset into.
    :returns: None
    """
        
    # Check to make sure all given dirs exist
    if not os.path.exists(sources_dir):
        raise OSError("The given sources directory does not exist.")

    if not os.path.exists(targets_dir):
        raise OSError("The given targets directory does not exist.")

    if not os.path.exists(output_dir):
        raise OSError("The given output directory does not exist.")

    # Look at every source image and its label
    source_labels = os.listdir(sources_dir)
    
    pbar = tqdm(source_labels)
    pbar.set_description("Adjusting Source Images")


    for label in source_labels:
        images = os.listdir(os.path.join(sources_dir, label))
        
        label_add = 0
        next_label_string = label
        prev_label_string = label

        # Find the next closest label 
        while not os.path.exists(os.path.join(targets_dir, next_label_string)) and not\
            os.path.exists(os.path.join(targets_dir, prev_label_string)):
            
            label_add += 1
            
            next_label_string = str(int(label) + label_add)
            prev_label_string = str(int(label) - label_add)

        target_label = label

        if os.path.exists(os.path.join(targets_dir, next_label_string)):
            target_label = next_label_string
        elif os.path.exists(os.path.join(targets_dir, prev_label_string)):
            target_label = prev_label_string
    
        
        target_images = os.listdir(os.path.join(targets_dir, target_label))

        for image in images:
            target_image = random.choice(target_images)

            source_path = os.path.join(sources_dir, label, image)
            target_path = os.path.join(targets_dir, target_label, target_image)
            
            # Match the source image to the chosen target
            matched_image = match_images(source_path, target_path)

        
            # Save the image in the output dir
            out_path = os.path.join(output_dir, label)

            if not os.path.exists(out_path):
                os.mkdir(out_path)

            out_path = os.path.join(out_path, image)
            
            io.imsave(out_path, matched_image)
        
        pbar.update(1)

    # Copy target images to output dir
    target_labels = os.listdir(targets_dir)
    
    print("\n") 
    pbar2 = tqdm(target_labels)
    pbar2.set_description("Copying Targets")

    for label in target_labels:

        label_out_dir = os.path.join(output_dir, label)
        
        if not os.path.exists(label_out_dir):
            os.mkdir(label_out_dir)

        target_imgs = os.listdir(os.path.join(targets_dir, label))

        for img in target_imgs:
            img_path = os.path.join(targets_dir, label, img)
            img_out_path = os.path.join(output_dir, label, img)

            copyfile(img_path, img_out_path)
        
        pbar2.update(1)

    print("\n")
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Match a given set of high quality images to a different set of images.")

    parser.add_argument('source_dir', help="The directory to get the images to match from.")
    parser.add_argument('target_dir', help="The directory to get the image to match the source images to.")
    parser.add_argument('output_dir', help="The directory to store the new dataset in")

    args = parser.parse_args()

    create_matched_dataset(args.source_dir, args.target_dir, args.output_dir) 
