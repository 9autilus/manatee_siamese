from __future__ import print_function
import os
import numpy as np
import random


import argparse
import sys # for flushing to stdout

from train import train_net
from test import test_net

seed = 1337 # for reproducibility
np.random.seed(seed)
random.seed(seed)

train_dir = '/home/govind/work/dataset/manatee/sketches_train'
test_dir  = '/home/govind/work/dataset/manatee/sketches_test'
test2_dir  = '/home/govind/work/dataset/manatee/sketches2_test'

default_store_model = 'model.h5'

# For debugging
#np.set_printoptions(threshold='nan')
#train_dir = '/home/govind/work/manatee/manatee_siamese/test_dataset/wrong_matches'
#test_dir  = '/home/govind/work/manatee/manatee_siamese/test_dataset/correct_matches'
#test2_dir  = '/home/govind/work/manatee/manatee_siamese/test_dataset/test2'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", help="train/test")
    parser.add_argument("--weights", help="weights to be stored/read")
    parser.add_argument("--test_mode", help="0:test vs test, 1: test vs train+test 2: test2 vs train+test")
    parser.add_argument("--epochs", help="#epochs while training")
    args = parser.parse_args()
    
    if not args.phase:
        parser.print_help()
        parser.error('Must specify phase (train/test)')
    elif args.phase not in ['train', 'test']:
        parser.print_help()
        parser.error('phase must be (train/test)')  
    
    if args.phase == 'train':
        args.epochs = int(args.epochs)
    else:
        args.test_mode = int(args.test_mode)
        if args.test_mode not in range(0,3):
            parser.print_help()
            parser.error('For testing, test_mode must be 0,1 or 2.')            

    if not args.weights:
        print('No weights specified. using default: ', default_store_model)
        args.weights = default_store_model
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    if args.phase == 'train':
        train_net(train_dir, args.weights, args.epochs)
    else:
        test_net(args)
    
    
