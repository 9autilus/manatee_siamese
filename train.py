from __future__ import print_function
from siamese_model import create_network
from dataset import Dataset
from keras.callbacks import ModelCheckpoint
import numpy as np
import random


class SolverWrapper():
    def __init__(self, imdb, weights, nb_epoch, train_dir):
        self.imdb = imdb
        self.weights_file = weights
        self.nb_epoch = nb_epoch
        self.input_dim = self.imdb.get_input_dim()
        self.train_dir = train_dir
        
        # network definition
        self.net = create_network(self.input_dim)


    def create_training_pairs(self, x):
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
            
            # Add labels
            if 1:
                labels += [1,0]
            else: # Conditionally add third pair
                while rand_idx in unusable_idx:
                    rand_idx = random.randrange(1, num_sample)
                unusable_idx += [rand_idx]    
                pairs += [[x[rand_idx], sample]]
                
                labels += [1, 0, 0]
        return np.array(pairs), np.array(labels)    
    

    def train_model(self):
        # the data, shuffled and split between train and test sets
        X, ID = self.imdb.load_sketches(self.train_dir)
        
        # Divide between training and validation set
        
        num_sketches = X.shape[0]
        num_sketches_val = (num_sketches * 30)/100
        num_sketches_train = num_sketches - num_sketches_val
        
        X_train = X[:num_sketches_train]
        X_val = X[num_sketches_train:]
        
        # create training+test positive and negative pairs
        print('Creating training pairs...')
        tr_pairs, tr_y = self.create_training_pairs(X_train)
        val_pairs, val_y = self.create_training_pairs(X_val)
        print('Done')

        # Create check point callback
        checkpointer = ModelCheckpoint(filepath=self.weights_file, 
            monitor='val_loss', verbose=1, save_best_only=True)
        
        # Train
        print('Training started ...')
        self.net.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  nb_epoch=self.nb_epoch,
                  validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y), 
                  callbacks=[checkpointer])
        print('Trainig complete. Saved model as: ', self.weights_file)            
    
def train_net(sketch_dir, weights, nb_epoch):
    imdb = dataset(train_dir=sketch_dir)
    
    sw = SolverWrapper(imdb, weights, nb_epoch, imdb.get_input_dim(), sketch_dir)
    
    sw.train_model()
   