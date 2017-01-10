#from __future__ import absolute_import
#from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import cv2


train_dir = r'G:\work\LiLab\DeepEyes\Datasets\d_manatee\sketches'
#test_dir  = r'G:\work\LiLab\DeepEyes\Datasets\d_manatee\test_set'
ht = 32
wd = 64

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def get_sketch(sketch_path):
    sketch = cv2.imread(sketch_path)
    if sketch is not None:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        sketch = cv2.resize(sketch, (wd, ht))
        sketch = (sketch/255.).astype('float32')
        return sketch.flatten()
    else:
        print('Unable to open ', sketch_path)
        return np.zeros(ht*wd).astype('float32')    
    
def load_training_data():
    train_sketch_names = ['U041.tif', 'U065.jpg','U232.tif', 'U310.jpg']
    #train_sketch_names = os.listdir(train_dir)
    
    print('Reading training data..')
    y_train = [x.split('.')[0] for x in train_sketch_names] # sketch names w/o extension
    X_train = np.empty([len(train_sketch_names), ht * wd], dtype='float32')
    for idx, sketch_name in enumerate(train_sketch_names):
        print(('\r{0:d}/{1:d} '.format(idx+1, len(train_sketch_names))), end='')
        X_train[idx] = get_sketch(os.path.join(train_dir, sketch_name))
    print('Done.')
    return X_train, y_train
    
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


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
    
    num_sample = x.shape[0]
    for idx, sample in enumerate(x):
        pairs += [[sample, sample]]
        rand_idx = idx
        while rand_idx == idx:
            rand_idx = random.randrange(1, num_sample)
        pairs += [[sample, x[rand_idx]]]
        
        labels += [1, 0]
    return np.array(pairs), np.array(labels)    

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() > 0.5))

if __name__ == '__main__':
    # the data, shuffled and split between train and test sets
    X_train, y_train = load_training_data()
    input_dim = X_train.shape[1]
    nb_epoch = 1#20

    # create training+test positive and negative pairs
    #digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train)

    # network definition
    base_network = create_base_network(input_dim)

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=1, #128,
              nb_epoch=nb_epoch)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
