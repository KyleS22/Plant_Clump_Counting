#! /usr/bin/env python

import argparse
import os
import sys
import numpy as np
import json
from voc import parse_voc_annotation
from unet import UNet as unet
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs, preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
import tensorflow as tf
import keras
from keras.models import load_model

import cv2

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_model(
    nb_class, 
    saved_weights_name, 
    net_input_height, 
    net_input_width
):
    infer_model = unet().create_unet_model((net_input_height, net_input_width, 3), nb_class)

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights.\n")
        infer_model.load_weights(saved_weights_name)
    else:
        tb = sys.exc_info()[2]
        raise ValueError(f"Pretrained weights at '{saved_weights_name}' not found!").with_traceback(tb)

    return infer_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    _, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\Predicting on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators 
    ###############################    
    '''train_generator = BatchGenerator(
        instances           = train_ints, 
        labels              = labels,        
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        #min_net_size        = config['model']['min_input_size'],
        #max_net_size        = config['model']['max_input_size'],
        net_input_height    = config['model']['net_input_height'],
        net_input_width     = config['model']['net_input_width'],
        shuffle             = config['train']['shuffle'],
        jitter              = config['train']['jitter'],
        allow_horizontal_flip = config['train']['allow_horizontal_flip'],
        allow_vertical_flip   = config['train']['allow_vertical_flip'],
        allow_90deg_rotate  = config['train']['allow_90deg_rotate'],
        min_resize_scale    = config['train']['min_resize_scale'],
        max_resize_scale    = config['train']['max_resize_scale'],
        hue_adjust          = config['train']['hue_adjust'],
        saturation_adjust   = config['train']['saturation_adjust'],
        brightness_adjust   = config['train']['brightness_adjust'],
        exposure_adjust     = config['train']['contrast_adjust'],
        norm                = normalize
    )'''
    
    valid_generator = BatchGenerator(
        instances           = valid_ints,  
        labels              = labels,        
        max_box_per_image   = max_box_per_image,
        batch_size          = config['valid']['batch_size'], # Make own validate one??? 
        #min_net_size        = config['model']['min_input_size'],
        #max_net_size        = config['model']['max_input_size'],
        net_input_height    = config['model']['net_input_height'],
        net_input_width     = config['model']['net_input_width'],
        shuffle             = config['valid']['shuffle'],
        jitter              = config['valid']['jitter'],
        allow_horizontal_flip = config['valid']['allow_horizontal_flip'],
        allow_vertical_flip   = config['valid']['allow_vertical_flip'],
        allow_90deg_rotate  = config['valid']['allow_90deg_rotate'],
        min_resize_scale    = config['valid']['min_resize_scale'],
        max_resize_scale    = config['valid']['max_resize_scale'],
        hue_adjust          = config['valid']['hue_adjust'],
        saturation_adjust   = config['valid']['saturation_adjust'],
        brightness_adjust   = config['valid']['brightness_adjust'],
        exposure_adjust     = config['valid']['contrast_adjust'],
        norm                = normalize
    )

    '''###############################
    #   Create the model 
    ###############################
    if os.path.exists(config['train']['saved_weights_name']): 
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))'''

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    '''multi_gpu = len(config['train']['gpus'].split(','))'''

    infer_model = create_model(
        nb_class            = len(labels), 
        saved_weights_name  = config['train']['saved_weights_name'],
        net_input_height    = config['model']['net_input_height'],
        net_input_width     = config['model']['net_input_width'],
    )

    '''###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], train_model)

    train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = config['train']['nb_epochs'], 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        workers          = 4,
        max_queue_size   = 8
    )

    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config['train']['saved_weights_name'])'''

    ###############################
    #   Run the prediction
    ###############################   
    '''# compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))'''
    
    outfolder = config['valid']['valid_save_folder']
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    for i in range(valid_generator.size()):
        filename = valid_generator.load_filename(i)
        raw_image = [valid_generator.load_image(i)]
        for img in raw_image:
            predict_image = preprocess_input(img, valid_generator.net_h, valid_generator.net_w)
            prediction = infer_model.predict(predict_image, batch_size=config['valid']['batch_size'], verbose=1)
            for input, output in zip(predict_image, prediction):
                output *= 255. # Get to 8-bit
                '''cv2.imshow('input', input)
                cv2.imshow('output', output)
                keyPressed = cv2.waitKey(0)
                cv2.destroyAllWindows()'''
            
                cv2.imwrite(os.path.join(outfolder, os.path.basename(filename[:filename.rfind('.')])+config['model']['mask_save_format']), output)

    '''results = infer_model.predict_generator(valid_generator, verbose=1)
    cv2.imshow('output', results[0]*255.)
    keyPressed = cv2.waitKey(0)
    cv2.destroyAllWindows()'''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='export U-Net predictions from saved model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
