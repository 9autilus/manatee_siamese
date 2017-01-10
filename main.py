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


train_dir = '/media/disk2/govind/work/dataset/manatee/sketches'
#test_dir  = '/media/disk2/govind/work/dataset/manatee/test_set'
ht = 64
wd = 128
nb_epoch = 2#20

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def get_sketch(sketch_path):
    sketch = cv2.imread(sketch_path)
    if sketch is not None:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        sketch = cv2.resize(sketch, (wd, ht))
        sketch = (sketch/255.).astype('float32')
    else:
        print('Unable to open ', sketch_path)
        return np.zeros([ht, wd]).astype('float32')    
    
def load_training_data():
    #train_sketch_names = ['U041.tif', 'U065.jpg','U232.tif', 'U310.jpg']
    train_sketch_names = os.listdir(train_dir)
    
    print('Reading training data..')
    y_train = [x.split('.')[0] for x in train_sketch_names] # sketch names w/o extension
    X_train = np.empty([len(train_sketch_names), ht, wd], dtype='float32')
    for idx, sketch_name in enumerate(train_sketch_names):
        print(('\r{0:d}/{1:d} '.format(idx+1, len(train_sketch_names))), end='')
        X_train[idx] = get_sketch(os.path.join(train_dir, sketch_name))
    print('Done.')
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    return X_train, y_train
    
def create_pairs(x):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    #n = min([len(digit_indices[d]) for d in range(10)]) - 1
    #for d in range(10):
    #    for i in range(n):
    #        z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
    #        pairs += [[x[z1], x[z2]]]
    #        inc = random.randrange(1, 10)
    #        dn = (d + inc) % 10
    #        z1, z2 = digit_indices[d][i], digit_indices[dn][i]
    #        pairs += [[x[z1], x[z2]]]
    #        labels += [1, 0]
    #return np.array(pairs), np.array(labels)
    
    # Add two pairs. First pair = same sketch
    # second pair = diferent sketches
    num_sample = x.shape[0]
    for idx, sample in enumerate(x):
        pairs += [[sample, sample]]
        rand_idx = idx
        while rand_idx == idx:
            rand_idx = random.randrange(1, num_sample)
        pairs += [[sample, x[rand_idx]]]
        
        labels += [1, 0]
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

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() > 0.5))

if __name__ == '__main__':
    # the data, shuffled and split between train and test sets
    X_train, y_train = load_training_data()
    input_dim = (1, X_train.shape[2], X_train.shape[3])

    # create training+test positive and negative pairs
    #digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train)

    # network definition
    base_network = create_base_network(input_dim)
    input_a = Input(shape=(input_dim))
    input_b = Input(shape=(input_dim))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    abs_diff = Lambda(get_abs_diff, output_shape = eucl_dist_output_shape)([processed_a, processed_b])
    flattened_weighted_distance = Dense(1, activation = 'sigmoid')(abs_diff)

    model = Model(input=[input_a, input_b], output = flattened_weighted_distance)     

    # train
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              nb_epoch=nb_epoch)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
