#from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import cv2
import argparse

train_dir = '/media/disk2/govind/work/dataset/manatee/sketches'
#test_dir  = '/media/disk2/govind/work/dataset/manatee/test_set'
ht = 64
wd = 128
nb_epoch = 50
default_store_model = 'model.h5'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", help="train/val")
    parser.add_argument("--weights", help="specify existing weights")
    args = parser.parse_args()
    
    if not args.phase:
        parser.print_help()
        parser.error('Must specify phase (train/val)')
    elif args.phase not in ['train', 'val']:
        parser.print_help()
        parser.error('phase must be (train/val)')

    if not args.weights:
        print('No weights specified. using default: ', default_store_model)
        args.weights = default_store_model
    return args

def get_sketch(sketch_path):
    sketch = cv2.imread(sketch_path)
    if sketch is not None:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        sketch = cv2.resize(sketch, (wd, ht))
        #inverting sketch for a black background
        sketch = (255 - sketch).astype('float32') 
        # Zero mean and Unit variance
        sketch = (sketch - sketch.mean())/sketch.var()
        return sketch
    else:
        print('Unable to open ', sketch_path, ' Skipping.')
        return None
    
def load_sketches():
    train_sketch_names = os.listdir(train_dir)
    train_sketch_names = train_sketch_names[-75:]
    
    print('Reading sketches from disk...')
    y_train = [x.split('.')[0] for x in train_sketch_names] # sketch names w/o extension
    X_train = np.empty([len(train_sketch_names), ht, wd], dtype='float32')
    for idx, sketch_name in enumerate(train_sketch_names):
        print(('\r{0:d}/{1:d} '.format(idx+1, len(train_sketch_names))), end='')
        sketch = get_sketch(os.path.join(train_dir, sketch_name))
        if sketch is not None:
            X_train[idx] = sketch
    print('Done.')
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    return X_train, y_train
    
def create_pairs(x):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []; labels = []
    
    # Add two pairs. First pair = same sketch
    # second pair = diferent sketches
    num_sample = x.shape[0]
    for idx, sample in enumerate(x):
        #Add first pair
        pairs += [[sample, sample]]
        unusable_idx = [idx]
        rand_idx = idx
        
        # Add second pair
        while rand_idx in unusable_idx:
            rand_idx = random.randrange(1, num_sample)
        unusable_idx += [rand_idx]    
        pairs += [[sample, x[rand_idx]]]
        
        # Conditionally add third pair
        if 1:
            labels += [1,0]
        else:
            while rand_idx in unusable_idx:
                rand_idx = random.randrange(1, num_sample)
            unusable_idx += [rand_idx]    
            pairs += [[x[rand_idx], sample]]
            
            labels += [1, 0, 0]
    return np.array(pairs), np.array(labels)    

def get_abs_diff( vects ):
    x, y = vects
    return K.abs( x - y )  

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1 
    
def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Convolution2D(48, 10, 10, activation='relu', border_mode='valid', input_shape=input_dim))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    seq.add(Convolution2D(128, 7, 7, activation='relu', border_mode='valid'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))    
    seq.add(Convolution2D(128, 4, 4, activation='relu', border_mode='valid'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))     
    seq.add(Convolution2D(256, 4, 4, activation='relu', border_mode='valid'))
    seq.add(Flatten())
    seq.add(Dense(4096, activation='sigmoid'))
    seq.add(Dense(1, activation='sigmoid'))
    return seq

def test_standard(model, tr_pairs, tr_y):
    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    print('* Standard Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    
def test_rank_based(model, X_train):
    print('Computing rank-based accuracy... ')
    ranks = [1, 5, 10, 20]
    accuracy = [0.] * len(ranks)

    num_samples = X_train.shape[0]
    num_pairs = (num_samples * (num_samples + 1)) / 2
    pairs = np.empty([num_pairs, 2, X_train[0].shape]);
    
    print('Formulating all possible pairs... ', end='')
    pair_idx = 0
    for i in range(num_samples):
        for j in range(i, num_samples):
            pairs[pair_idx] = np.array([[X_train[i], X_train[j]]])
            pair_idx += 1
    print('Done.')
    
    print('Predicting... ', end='')
    scores =  model.predict([pairs[:, 0], pairs[:, 1]]) 
    print('Done.')
    
    print('Analyzing scores... ', end='')
    score_table = np.zeros([num_samples, num_samples]).astype('float32')
    idx = 0
    for i in range(num_samples):
        for j in range(i, num_samples):
            score_table[i, j] = scores[idx]
            idx = idx + 1
    score_table += np.transpose(score_table)
    for i in range(num_samples):
        score_table[i][i] /= 2.
        
    sorted_idx = np.argsort(score_table, axis=1)
    
    for r, rank in enumerate(ranks):
        num_found = 0
        for i in range(num_samples):
            if i in sorted_idx[i][0:rank]:
                num_found += 1
        accuracy[r] = (100 * num_found)/float(num_samples)
        
    print('Rank based accuracy:')
    for i in range(len(ranks)):
        print('Rank {0:3d} : {1:2.2f}%'.format(ranks[i], accuracy[i]))
    
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() > 0.5))

if __name__ == '__main__':
    args = parse_arguments()
    
    # the data, shuffled and split between train and test sets
    X_train, y_train = load_sketches()
    input_dim = (1, X_train.shape[2], X_train.shape[3])

    # create training+test positive and negative pairs
    tr_pairs, tr_y = create_pairs(X_train)

    # network definition
    base_network = create_base_network(input_dim)
    input_a = Input(shape=(input_dim))
    input_b = Input(shape=(input_dim))

    # because we re-use the same instance `base_network`,
    # the weights of the network will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    abs_diff = Lambda(get_abs_diff, output_shape = eucl_dist_output_shape)([processed_a, processed_b])
    flattened_weighted_distance = Dense(1, activation = 'sigmoid')(abs_diff)

    model = Model(input=[input_a, input_b], output = flattened_weighted_distance)     

    # Optimizer
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    
    if args.phase == 'train':
        # Train
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  nb_epoch=nb_epoch)
        print('Saving model as: ', args.weights)
        model.save_weights(args.weights) # Write learned weights on disk    
    else:
        # Reuse pre-trained weights
        print('Reading weights from disk: ', args.weights)
        model.load_weights(args.weights)
        
    test_standard(model, tr_pairs, tr_y)
    test_rank_based(model, X_train)
    
    
