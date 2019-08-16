"""
File Name: train_runner.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: Script to train the different models

"""

import argparse
import os
import json
import sys

from datetime import datetime

from fourier_based_model import FourierTransformModel
from glcm_based_model import GLCMModel 
from lbph_based_model import 

def train_model(save_dir, model_name, training_data_dir, model_type,
        feature_type, validation_data_dir=None):
    """
    Train the chosen model.
    
    :param save_dir:            The directory to save the trained model in.  It must not
                                exist beforehand.
    :param model_name:          The name of the model, for file naming purposes/
    :param training_data_dir:   The directory to get training data from
    :param model_type:          The type of model to use.  Must be one of 'KNN', 'SVC',
                                'GNB'
    :param feature_type:        The type of features to use for the model.
                                Must be one of 'FFT', 'LBPH', 'GLCM'
    :param validation_data_dir: The directory to get the validation images
                                from
    :returns: None.     The model will be trained and saved in the appropriate
                        directory.
    """
    
    if feature_type is None:
        raise IOError("The chosen feature type is invalid.  Please choose one
                of 'FFT', 'LBPH', 'GLCM'")
    elif feature_type == "FFT":
        model = FourierTransformModel(model_type=model_type, save_path=save_dir)
    elif featire_type == "GLCM":
        model = GLCMModel(model_type=model_type, save_path=save_dir)
    elif feature_type == "LBPH":
        model = LBPHModel(model_type=model_type, save_path=save_dir)

    model.fit(training_data_dir)

    

def _save_config_json(out_dir, args, start_time=None):
    """
    Save the configurations of this run to a JSON file
    
    :param out_dir: The directory to store the save file in
    :param args: The argparse arguments used to run the program
    :param start_time=None: The Start time of the program.  Default is none
    :returns: None
    """
    end_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    config = {}
    
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    config['start_time'] = start_time
    config['end_time'] = end_time
   
    out_path = os.path.join(out_dir, 'config.json')

    with open(out_path, 'w') as config_json:
        json.dump(config, config_json)


 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a counting model.") 

    parser.add_argument('--model_save_dir', default=None, help="The directory to save model files to.  The model will not be saved if not specified.")
    parser.add_argument('training_data_dir', help='The directory to get training data from.')
    parser.add_argument('model_type', help="The type of model to use.  Can be
                                      one of 'KNN', 'SVC', GNB'")
    parser.add_argument('feature_type', help="The type of image features to
            use.  Must be one of 'FFT', 'LBPH', 'GLCM'")
    parser.add_argument('--validation_data_dir', help='The directory to get validation data from.')
    parser.add_argument('--model_name', default="model", help="The name of the model.")

    args = parser.parse_args()
    
    start_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    
    if args.model_save_dir is not None and os.path.exists(args.model_save_dir):
        print("The model save path you have chosen already exists.  Please choose another.")
        sys.exit(1)

    train_model(args.model_save_dir, args.model_name, args.training_data_dir,
            args.model_type, args.feature_type, validation_data_dir=args.validation_data_dir)


    _save_config_json(args.model_save_dir, args, start_time=start_time)
