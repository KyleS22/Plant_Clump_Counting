"""
File Name: train_runner.py

Authors: Kyle Seidenthal

Date: 23-07-2019

Description: Script to train the CNN

"""

import argparse
import os
import json
import sys

from datetime import datetime

from model import LossHistory, EncoderCountingModel

def train_model(save_dir, model_name, training_data_dir, batch_size, num_epochs, validation_data_dir=None):
    """
     Train the encoder model and save it 
    
    :param save_dir: The directory name to save the model in
    :param model_name: The name of the model (for file naming purposes)
    :param training_data_dir: The directory to get training images from
    :param batch_size: The batch size to use
    :param num_epochs: The number of epochs to train for
    :param validation_data_dir: The directory to get validation data from
    :returns: None
    """

    model = EncoderCountingModel(save_dir, name=model_name)
    
    model.compile()

    history = model.train(training_data_dir, batch_size, num_epochs, validation_data_dir=validation_data_dir)
    
    history.save(save_dir)

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
    parser.add_argument('--validation_data_dir', help='The directory to get validation data from.')
    parser.add_argument('--batch_size', default=8, type=int, help="The batch size to use when training.")
    parser.add_argument('--num_epochs', default=100, type=int, help="The number of epochs to use when training.")
    parser.add_argument('--model_name', default="model", help="The name of the model.")

    args = parser.parse_args()
    
    start_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    
    if args.model_save_dir is not None and os.path.exists(args.model_save_dir):
        print("The model save path you have chosen already exists.  Please choose another.")
        sys.exit(1)

    train_model(args.model_save_dir, args.model_name, args.training_data_dir, args.batch_size, args.num_epochs, validation_data_dir=args.validation_data_dir)


    _save_config_json(args.model_save_dir, args, start_time=start_time)
