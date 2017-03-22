from __future__ import print_function
import os

seed = 1337  # for reproducibility
import numpy as np
np.random.seed(seed)        # Seed Numpy
import random               # Seed random
random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)    # Seed Tensor Flow

# Use theano dimension ordering
from keras import backend as K
K.set_image_dim_ordering('th')

import argparse
from train import train_net
from test import test_net

default_store_model = 'model.h5'

# For printing complete numpy array while debugging
#np.set_printoptions(threshold='nan')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", help="train/test")
    parser.add_argument("--model", help="model to be stored/read")
    parser.add_argument("--test_mode", help="0:test vs test, 1: test vs train+test 2: test2 vs train+test")
    parser.add_argument("--epochs", help="#epochs while training")
    parser.add_argument("--retrain", help="Continue training existing model")
    parser.add_argument("--initial_epoch", help="Starting epoch. Used when --retrain")
    parser.add_argument("--gpu", help="list of GPUs, e.g. [0,1,3]")


    args = parser.parse_args()
    if not args.phase:
        parser.print_help()
        parser.error('Must specify phase (train/test)')
    elif args.phase not in ['train', 'test']:
        parser.print_help()
        parser.error('phase must be (train/test)')

    if args.phase == 'train':
        if args.test_mode:
            print('Ignoring test_mode parameter for training.')
        args.epochs = int(args.epochs)

        if args.retrain:
            if not args.model:
                parser.print_help()
                parser.error('Must specify --model with --retrain')
            if not args.initial_epoch:
                parser.print_help()
                parser.error('Must specify --initial_epoch with --retrain')
            else:
                args.initial_epoch = int(args.initial_epoch)
    else:
        args.test_mode = int(args.test_mode)
        if args.test_mode not in range(0, 3):
            parser.print_help()
            parser.error('For testing, test_mode must be 0,1 or 2.')

    if not args.model:
        print('No model specified. using default: ', default_store_model)
        args.model = default_store_model

    if not args.gpu:
        args.gpu = '0'

    return args

if __name__ == '__main__':
    args = parse_arguments()

    # Limit execution on certain GPU/GPUs
    gpu_id = args.gpu  # Comma separated string of GPU IDs to be used e.g. '0, 1, 2, 3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    common_cfg_file = os.path.join('configure', 'common.json')
    train_cfg_file = os.path.join('configure', 'train.json')
    test_cfg_file = os.path.join('configure', 'test.json')

    if args.phase == 'train':
        train_net(
            common_cfg_file, train_cfg_file, args.model, args.epochs,\
            args.retrain, args.initial_epoch)
    else:
        test_net(common_cfg_file, test_cfg_file, args.test_mode, args.model)
