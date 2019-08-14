#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

def _main_(args):
    config_path = args.conf
    dataset     = args.dataset

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################  
    valid_ints, labels = parse_voc_annotation(
        config[dataset][dataset+'_annot_folder'], 
        config[dataset][dataset+'_image_folder'], 
        config[dataset]['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = config[dataset]['shuffle'],
        jitter              = config[dataset]['jitter'],
        allow_horizontal_flip = config[dataset]['allow_horizontal_flip'],
        allow_vertical_flip   = config[dataset]['allow_vertical_flip'],
        allow_90deg_rotate  = config[dataset]['allow_90deg_rotate'],
        min_resize_scale    = config[dataset]['min_resize_scale'],
        max_resize_scale    = config[dataset]['max_resize_scale'],
        hue_adjust          = config[dataset]['hue_adjust'],
        saturation_adjust   = config[dataset]['saturation_adjust'],
        brightness_adjust   = config[dataset]['brightness_adjust'],
        exposure_adjust     = config[dataset]['contrast_adjust'],
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    infer_model = load_model(config['train']['saved_weights_name'])
    
    if args.summary:
        print(infer_model.summary())

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))           

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')    
    argparser.add_argument('-d', '--dataset', help='whether to use the \'train\', \'valid\', or \'test\', paths in the configuration file. (default=valid)', type=str, default='valid')
    argparser.add_argument('--summary', help='print the model summary before evaluation', action='store_true')
    
    args = argparser.parse_args()
    _main_(args)
