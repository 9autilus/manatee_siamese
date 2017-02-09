#from __future__ import absolute_import
from __future__ import print_function
seed = 1337 # for reproducibility

import os
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import cv2
import argparse
import sys # for flushing to stdout

train_dir = '/media/disk2/govind/work/dataset/manatee/sketches_train'
test_dir  = '/media/disk2/govind/work/dataset/manatee/sketches_test'
test2_dir = '/media/disk2/govind/work/dataset/manatee/sketches2_test'
ht = 128
wd = 256
default_store_model = 'model.h5'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", help="train/test")
    parser.add_argument("--weights", help="specify existing weights")
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
        if args.test_mode:
            print('Ignoring test_mode parameter for training.')
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
    
    
def attach_pairs(sketch_dir):
    if not os.path.exists(sketch_dir):
        print('The sketch directory {0:s} does not exist'.format(sketch_dir))
        return None, None, None, None    
    
    sketch_names = os.listdir(sketch_dir)
    random.shuffle(sketch_names) # Shuffle
    
    #sketch_names = sketch_names[-100:] # Enable for debugging purpose
    
    # Divide between training and validation set
    num_sketches_train = (len(sketch_names) * 70)/100
    
    train_sketches = sketch_names[:num_sketches_train]
    val_sketches = sketch_names[num_sketches_train:]
    
    # Create training pairs
    train_pairs = []
    train_labels = [1, 0] * len(train_sketches)
    for i, sketch_name in enumerate(train_sketches):
        train_pairs.append([sketch_name, sketch_name])
        
        rand_idx = i
        while rand_idx == i:
            rand_idx = random.randrange(0, len(train_sketches))
        train_pairs.append([sketch_name, train_sketches[rand_idx]])
    
    # Create validation pairs
    val_pairs = []
    val_labels = [1, 0] * len(val_sketches)
    for sketch_name in val_sketches:
        val_pairs.append([sketch_name, sketch_name])
        
        rand_idx = i
        while rand_idx == i:
            rand_idx = random.randrange(0, len(val_sketches))        
        
        val_pairs.append([sketch_name, val_sketches[rand_idx]])   
    
    return train_pairs, train_labels, val_pairs, val_labels
    
def get_train_batch(sketch_dir, pairs, labels, batch_size):
    # batch size must be even number
    if batch_size % 2 != 0:
        print('Error: batch size must be an even number')
        exit(0)
    
    X_l = np.zeros((batch_size, 1, ht, wd))
    X_r = np.zeros((batch_size, 1, ht, wd))
    y   = np.array([1, 0] * (batch_size/2))
    
    src_idx = 0
    dst_idx = 0
    while True:
        sketch1_name = pairs[src_idx][0]
        sketch2_name = pairs[src_idx + 1][1]
        sketch1 = get_sketch(os.path.join(sketch_dir, sketch1_name))
        sketch2 = get_sketch(os.path.join(sketch_dir, sketch2_name))
    
        X_l[dst_idx]     = sketch1.reshape(1, ht, wd)
        X_r[dst_idx]     = sketch1.reshape(1, ht, wd)
        X_l[dst_idx + 1] = sketch1.reshape(1, ht, wd)
        X_r[dst_idx + 1] = sketch2.reshape(1, ht, wd)
    
        src_idx += 2
        dst_idx += 2
        
        if src_idx >= len(pairs):
            src_idx = 0
        
        if dst_idx >= batch_size:
            dst_idx = 0
            yield [X_l, X_r], y   
            
def get_val_batch(sketch_dir, pairs, labels, batch_size):
    # batch size must be even number
    if batch_size % 2 != 0:
        print('Error: batch size must be an even number')
        exit(0)
    
    X_l = np.zeros((batch_size, 1, ht, wd))
    X_r = np.zeros((batch_size, 1, ht, wd))
    y   = np.array([1, 0] * (batch_size/2))
    
    src_idx = 0
    dst_idx = 0
    while True:
        sketch1_name = pairs[src_idx][0]
        sketch2_name = pairs[src_idx + 1][1]
        sketch1 = get_sketch(os.path.join(sketch_dir, sketch1_name))
        sketch2 = get_sketch(os.path.join(sketch_dir, sketch2_name))
    
        X_l[dst_idx]     = sketch1.reshape(1, ht, wd)
        X_r[dst_idx]     = sketch1.reshape(1, ht, wd)
        X_l[dst_idx + 1] = sketch1.reshape(1, ht, wd)
        X_r[dst_idx + 1] = sketch2.reshape(1, ht, wd)
    
        src_idx += 2
        dst_idx += 2
        
        if src_idx >= len(pairs):
            src_idx = 0
        
        if dst_idx >= batch_size:
            dst_idx = 0
            yield [X_l, X_r], y    
    
def load_sketches(sketch_dir):
    if not os.path.exists(sketch_dir):
        print('The sketch directory {0:s} does not exist'.format(sketch_dir))
        return None, None

    sketch_names = os.listdir(sketch_dir)
    random.shuffle(sketch_names) # Shuffle
    
    #sketch_names = sketch_names[-100:] # Enable for debugging purpose
    
    if len(sketch_names) < 1:
        print('Found only {0:d} sketches in the sketch directory: {1:s}'.\
            format(len(sketch_names), sketch_dir),
            'What are you trying to do? Aborting for now.')
        exit(0)    
        
    print('Reading sketches from {0:s}'.format(sketch_dir))
    ID = [x.split('.')[0] for x in sketch_names] # sketch names w/o extension
    X = np.empty([len(sketch_names), ht, wd], dtype='float32')
    for idx, sketch_name in enumerate(sketch_names):
        print(('\r{0:d}/{1:d} '.format(idx+1, len(sketch_names))), end='')
        sketch = get_sketch(os.path.join(sketch_dir, sketch_name))
        if sketch is not None:
            X[idx] = sketch
    print('Done.')
    
    X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
    return X, ID
    
def dump_sketch_pairs(sketch_dir, sketch_pairs, prefix):
    num_pairs = len(sketch_pairs)
    labels = np.zeros(num_pairs) # Won't be used
    batch_size = 2
    
    print(sketch_pairs)
    my_generator = get_train_batch(sketch_dir, sketch_pairs, labels, batch_size)
    
    for i in range(0, num_pairs, 2):
        X, y = next(my_generator)
    
        sketch = np.concatenate((X[0][0], X[0][1]), axis=1)
        sketch = sketch.reshape((2*ht, wd))
        sketch = sketch - np.min(sketch) # bring lower limit to 0
        sketch = sketch * (255. / np.max(sketch))
        sketch = sketch.astype('uint8')
        cv2.imwrite(prefix + str(i) + '_0.jpg', sketch)
        
        sketch = np.concatenate((X[1][0], X[1][1]), axis=1)
        sketch = sketch.reshape((2*ht, wd))
        sketch = sketch - np.min(sketch) # bring lower limit to 0
        sketch = sketch * (255. / np.max(sketch))
        sketch = sketch.astype('uint8')        
        cv2.imwrite(prefix + str(i) + '_1.jpg', sketch)        


def test_generators(sketch_dir, train_pairs, train_labels, val_pairs, val_labels, batch_size, num_train, num_val):
    epsilon = 0.001;
    num_train_batches = np.ceil(num_train/float(batch_size)).astype('int32')
    num_val_batches = np.ceil(num_train/float(batch_size)).astype('int32')
    
    gen = get_train_batch(sketch_dir, train_pairs, train_labels, batch_size)
    
    print('Testing training-set labels:')
    compare_labels(gen, train_pairs, train_labels, num_train_batches, batch_size)
    print('Testing validation-set labels:')
    compare_labels(gen, val_pairs, val_labels, num_val_batches, batch_size)    
    
    
def compare_labels(gen, pairs, labels, num_batches, batch_size):
    num_matches = 0
    for batch_id in range(num_batches):
        [x_l, x_r], y = gen.next();
        print('\rbatch_id: {0:d}/{1:d}'.format(batch_id, num_batches), end=''); sys.stdout.flush()
        
        for i in range(batch_size):
            num_matches += (labels[i] == np.array_equal(x_l[i], x_r[i]))
                
    print('\nLabels match: {0:.1f}%'.format(100 * num_matches/float(num_batches*batch_size)))
        
        
def create_training_pairs(x):
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
    
def create_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Convolution2D(48, 10, 10, activation='relu', border_mode='valid', input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(128, 7, 7, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))    
    model.add(Convolution2D(128, 4, 4, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))     
    model.add(Convolution2D(256, 4, 4, activation='relu', border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    
    input_a = Input(shape=(input_dim))
    input_b = Input(shape=(input_dim))

    # because we re-use the same instance `model`,
    # the weights of the network will be shared across the two branches
    processed_a = model(input_a)
    processed_b = model(input_b)

    abs_diff = Lambda(get_abs_diff, output_shape = eucl_dist_output_shape)([processed_a, processed_b])
    flattened_weighted_distance = Dense(1, activation = 'sigmoid')(abs_diff)

    model = Model(input=[input_a, input_b], output = flattened_weighted_distance)     

    # Optimizer
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])    
    
    return model
       
def perform_testing(model, X1, ID1, X2=None, ID2=None):
    test_single_source = (X2 is None) or (X1 is None)

    print('Computing rank-based accuracy... ')
    num_rows = X1.shape[0]
    
    if test_single_source:
        X2 = X1; ID2 = ID1
        num_cols = num_rows
        num_pairs = (num_rows * (num_rows + 1)) / 2
    else:
        num_cols = X2.shape[0]
        num_pairs = num_rows * num_cols
    
    ranks = sorted([1, 5, 10, 20])
    ranks = [rank for rank in ranks if rank <= num_cols] # filter bad ranks
    accuracy = [0.] * len(ranks)          
    
    # To tackle memory constraints, process pairs in small sized batchs
    batch_size = 128;    # Num pairs in a batch
    scores = np.empty(num_pairs).astype('float32') # List to store scores of ALL pairs     
    score_table = np.zeros([num_rows, num_cols]).astype('float32')
    
    if test_single_source:
        pairs = np.empty([batch_size, 2] + [i for i in X1[0].shape]);
        pair_count = 0; counter = 0;
        for i in range(num_rows):
            for j in range(i, num_cols):
                pairs[pair_count] = np.array([[X1[i], X1[j]]])
                pair_count += 1
                
                if (pair_count == batch_size):
                    print('\r Predicting pair {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                    batch_scores = model.predict([pairs[:, 0], pairs[:, 1]], batch_size=batch_size)
                    scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                    pair_count = 0
                    counter += batch_size
                    
        # Last remaining pairs which couldn't get processed in main loop
        if pair_count > 0:
            print('\r Predicting pair {0:d}/{1:d}'.format(num_pairs, num_pairs), end=''); sys.stdout.flush()
            batch_scores = model.predict([pairs[:pair_count, 0], pairs[:pair_count, 1]])
            scores[counter:(counter+pair_count)] = batch_scores[:, 0]
            
        print('Analyzing scores... ', end='')
        idx = 0
        for i in range(num_rows):
            for j in range(i, num_cols):
                score_table[i, j] = scores[idx]
                idx = idx + 1
        score_table += np.transpose(score_table)
        for i in range(num_rows):
            score_table[i][i] /= 2.
    else:   
        pairs = np.empty([batch_size, 2] + [i for i in X1[0].shape]);
        pair_count = 0; counter = 0;
        for i in range(num_rows):
            for j in range(num_cols):
                pairs[pair_count] = np.array([[X1[i], X2[j]]])
                pair_count += 1
                
                if (pair_count == batch_size):
                    print('\r Predicting pair {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                    batch_scores = model.predict([pairs[:, 0], pairs[:, 1]], batch_size=batch_size)
                    scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                    pair_count = 0
                    counter += batch_size
                    
        # Last remaining pairs which couldn't get processed in main loop
        if pair_count > 0:
            print('\r Predicting pair {0:d}/{1:d}'.format(num_pairs, num_pairs), end=''); sys.stdout.flush()
            batch_scores = model.predict([pairs[:pair_count, 0], pairs[:pair_count, 1]])
            scores[counter:(counter+pair_count)] = batch_scores[:, 0]
            
        print('Analyzing scores... ', end='')
        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                score_table[i, j] = scores[idx]
                idx = idx + 1         
    
    # Parse score table and generate accuracy metrics
    eval_score_table(score_table, ID1, ID2)        
        
def eval_score_table(score_table, row_IDs, col_IDs):  
    row_IDs = np.array(row_IDs)
    col_IDs = np.array(col_IDs)

    num_col = col_IDs.shape[0]
    num_row = row_IDs.shape[0]

    # verify score_table size against row_IDs and col_IDs
    #&&&
      
    ranks = sorted([1, 5, 10, 20])
    # verify ranks agains score_table size 
    #ranks = [rank for rank in ranks if rank <= num_samples] # filter bad ranks
    accuracy = [0.] * len(ranks)    
      
    sorted_idx = np.argsort(score_table, axis=1)
  
    for r, rank in enumerate(ranks):
        num_matches = 0
        for i in range(num_row):
            sorted_row_ids = col_IDs[sorted_idx[i]]
            if row_IDs[i] in sorted_row_ids[-rank:]:
                num_matches += 1
        accuracy[r] = (100 * num_matches)/float(num_row)
        
    print('Rank based accuracy:')
    for i in range(len(ranks)):
        print('Top {0:3d} : {1:2.2f}%'.format(ranks[i], accuracy[i]))
        
def train_net(args):
    sketch_dir = train_dir
    train_pairs, train_labels, val_pairs, val_labels = attach_pairs(sketch_dir)
    #dump_sketch_pairs(sketch_dir, train_pairs, 'tr_')
    #dump_sketch_pairs(sketch_dir, val_pairs, 'val_')
    
    batch_size = 32
    num_pairs = len(train_pairs)
    # Make samples_per_epoch a multiple of 32 to avoid warning:
    # "Epoch comprised more than `samples_per_epoch` samples" during training
    samples_per_epoch = 32 * np.ceil(num_pairs/32.)    
    
    test_generators(sketch_dir, train_pairs, train_labels, val_pairs, val_labels, 
        batch_size, samples_per_epoch, len(val_pairs))
        
    return

    # network definition
    input_dim = (1, ht, wd)
    model = create_network(input_dim)    

    # Create check point callback
    checkpointer = ModelCheckpoint(filepath=args.weights, 
        monitor='val_loss', verbose=1, save_best_only=True)
        
    # Train
    model.fit_generator(get_train_batch(sketch_dir, train_pairs, train_labels, batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=args.epochs,
        validation_data=get_val_batch(sketch_dir, val_pairs, val_labels, batch_size),
        nb_val_samples=len(val_pairs),
        callbacks=[checkpointer])    
    
    #model.fit_generator(gen_train, samples_per_epoch=len(train_pairs),
    #          nb_epoch=nb_epoch,
    #          validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y), 
    #          callbacks=[checkpointer])
    print('Trainig complete. Saved model as: ', args.weights)
    #model.save_weights(args.weights) # Write learned weights on disk     
    
def test_net(args):
    input_dim = (1, ht, wd)
    # network definition
    model = create_network(input_dim)

    # Reuse pre-trained weights
    print('Reading weights from disk: ', args.weights)
    model.load_weights(args.weights)
    
    if args.test_mode == 0:
        X, ID = load_sketches(test_dir)
        perform_testing(model, X, ID)
    elif args.test_mode == 1:
        X1, ID1 = load_sketches(test_dir)
        X2, ID2 = load_sketches(train_dir)
        X2 = np.concatenate((X1, X2), axis=0);
        ID2 = np.concatenate((ID1, ID2), axis=0);
        perform_testing(model, X1, ID1, X2, ID2)
    elif args.test_mode == 2:
        X1, ID1 = load_sketches(test2_dir)
        X2_1, ID2_1 = load_sketches(train_dir)
        X2_2, ID2_2 = load_sketches(test_dir)
        X2 = np.concatenate((X2_1, X2_2), axis=0);
        ID2 = np.concatenate((ID2_1, ID2_2), axis=0);
        perform_testing(model, X1, ID1, X2, ID2)
    
    
if __name__ == '__main__':
    args = parse_arguments()
    
    if args.phase == 'train':
        train_net(args)
    else:
        test_net(args)
    
    
