#!/usr/bin/python3

import os,sys
import json
import argparse

import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='gcn.json', help='Path to config contianing model parameters')
#parser.add_argument('-m', '--model', default='GCN', help='Name of the model to be used')
args = parser.parse_args()

model = None

# read the input config file
with open(args.config, 'r') as cfg:
    model      = cfg['model']
    input_args = cfg['input_args']
    
    # create a folder with the following items:
    #   - training.info
    #     - hash of the config map
    #     - date and timestamp
    #     - system, CUDA info
    #   - training.out: output of the training
    #   - training.log: log of the training
    #   - training.err: training error
    #   - weights/    : directory containing weights of the model
    #   - 
