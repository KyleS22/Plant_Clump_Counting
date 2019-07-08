"""
File Name: 

Authors: Kyle Seidenthal

Date: 08-07-2019

Description: A module containing functions that create a training and testing dataset

"""

import os

from sklearn.model_selection import train_test_split

import numpy as np
import shutil


def _get_images_and_labels(data_dir, labels_to_ignore=[]):
    """
    Get a list of the image names and their corresponding labels from the given directory.  Assumes that the directory
    given contains a folder for each label, each of which contain all of the images that match that label.
    
    :param data_dir: The directory to get images from
    :param labels_to_ignore=[]: A list of the names of the directories (matching the labels) to leave out of the process
    :returns: A tuple containing two lists, one with the names of each image, the other containing the labels that
              correspond to each image
    """
    
    labels = os.listdir(data_dir)

    images = []
    image_labels = []

    for label in labels:
        if not label in labels_to_ignore:
            for image in os.listdir(os.path.join(data_dir, label)):
                images.append(image)
                image_labels.append(label)

    return images, image_labels


def _create_train_and_test_dirs(train_dir, test_dir, overwrite=False):
    """
    Create the directories for training and testing data.  Note that the overwrite option allows you to delete the given
    directories if they already exist.  USE WITH CAUTION!
    
    :param train_dir: The directory to store the training data in
    :param test_dir: The directory to store the testing data in
    :param overwrite: Wheter or not to overwrite existing training and testing data in the given directories.  Default
                      is FALSE
    :returns: None
    """
        
    if os.path.exists(train_dir) or os.path.exists(test_dir):
        
        if overwrite:
            shutil.rmtree(train_dir)
            shutil.rmtree(test_dir)
            
            try:
                os.mkdir(train_dir)
                os.mkdir(test_dir)

            except:
                raise Exception("Could not make train and test dirs")

        else:
            raise Exception("The train and test directories already exist")

def _split_train_and_test_data(data_dir, train_dir, test_dir, images, image_labels, test_size=0.2):
    """
    Move the images from the data directory into the training and testing directories
    
    :param data_dir: The directory to get the images from
    :param train_dir: The directory to store training images in
    :param test_dir: The directory to store testing images in
    :param images: A list of the image names
    :param image_labels: A list of the labels for each of the images
    :param test_size: The size of the test set.  Default is 0.2
    :returns: None
    """
    
    X_train, X_test, y_train, y_test = train_test_split(images, image_labels, test_size=test_size, random_state=42)
    
    # Copy the images for the training set
    for image, label in zip(X_train, y_train):
        new_path = os.path.join(train_dir, label)
        old_path = os.path.join(data_dir, label, image)

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        shutil.copy(old_path, new_path)

    # Copy the images for the testing set
    for image, label in zip(X_test, y_test):
        new_path = os.path.join(test_dir, label)
        old_path = os.path.join(data_dir, label, image)

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        shutil.copy(old_path, new_path)

def create_train_and_test_split(data_dir, train_dir, test_dir, labels_to_ignore=[], test_size=0.2,  overwrite_old_data=False):
    """
    Split the data in the given directory into a train and test set by moving the images into the corresponding folders.
    
    :param data_dir: The directory containing all of the data, where images are stored in a folder with the name of
                     their label
    :param train_dir: The directory to store the training images in
    :param test_dir: The directory to store the test data in.  Will be created if it does not exist, and overwritten if
                     overwrite_old_data is True
    :param labels_to_ignore: A list of strings that are the names of the labels to ignore from data_dir
    :param test_size=0.2: The size of the test set ranging from 0 to 1 (percentage)
    :param overwrite_old_data: Whether or not to overwrite old train test split data.  THIS WILL DELETE ANY DATA
                             IN train_dir AND test_dir.  USE WITH CAUTION!
    :returns: A list of the image names and a list of the labels corresponding to each image
    """
     
    
    images, labels = _get_images_and_labels(data_dir, labels_to_ignore)
    _create_train_and_test_dirs(train_dir, test_dir, overwrite_old_data)
    _split_train_and_test_data(data_dir, train_dir, test_dir, images, labels, test_size=test_size)

    return images, labels
