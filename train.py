from __future__ import print_function
from siamese_model import create_network
from dataset import Dataset
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model
import numpy as np
import random
import os
import json

seed = 1337

class Reset_Random(Callback):
    # def on_train_begin(self, logs={}):
    #     return
    # 
    # def on_train_end(self, logs={}):
    #     return
 
    def on_epoch_begin(self, epoch, logs={}):
        np.random.seed(seed)  
        return
 
    # def on_epoch_end(self, epoch, logs={}):
    #     return
    # 
    # def on_batch_begin(self, batch, logs={}):
    #     return
    # 
    # def on_batch_end(self, batch, logs={}):
    #     self.losses.append(logs.get('loss'))
    #     return   

class SolverWrapper():
    def __init__(self, imdb, model_file, nb_epoch, batch_size, train_dir,
                 retrain, initial_epoch):
        self.imdb = imdb
        self.model_file = model_file
        self.nb_epoch = nb_epoch
        self.input_dim = self.imdb.get_input_dim()
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.retrain = retrain
        self.initial_epoch = initial_epoch
        
        # network definition
        if self.retrain:
            # Use pre-trained model_file
            print('Reading model from disk: ', model_file)
            self.net = load_model(model_file)
        else:
            self.net = create_network(self.input_dim)
            self.initial_epoch = 0

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
    
    def _dump_history(self, history, val_set_present, log_file_name):
        f = open(log_file_name, "w")

        train_acc = history['acc']
        train_loss = history['loss']

        if val_set_present:
            val_acc = history['val_acc']
            val_loss = history['val_loss']
        else:
            val_acc = [-1] * len(train_acc)
            val_loss = val_acc

        f.write('Epoch  Train_loss  Train_acc  Val_loss  Val_acc  \n')

        for i in range(len(train_acc)):
            f.write('{0:d} {1:.2f} {2:.2f}% {3:.2f} {4:.2f}%\n'.format(
                i, train_loss[i], 100 * train_acc[i], val_loss[i], 100 * val_acc[i]))

        print('Dumped history to file: {0:s}'.format(log_file_name))

        
    def train_model(self):
        batch_size = self.batch_size

        # Make num_train_sample a multiple of batch_size to avoid warning:
        # "Epoch comprised more than `num_train_sample` samples" during training
        num_train_sample = batch_size * np.ceil(self.imdb.get_num_train_sample() / float(batch_size)).astype('int32')
        num_val_sample = batch_size * np.ceil(self.imdb.get_num_val_sample() / float(batch_size)).astype('int32')

        # For debugging purpose
        if 1:
            self.imdb.validate_dataset(self.batch_size)

        # network definition
        model = create_network(self.input_dim)
        
        # Create check point callback
        checkpointer = ModelCheckpoint(filepath=self.model_file,
                                       monitor='val_loss', verbose=1, save_best_only=True)
                                       
        # reset_random = Reset_Random()

        ## Reduce sample-count for debugging
        # num_train_sample = 2* self.batch_size
        # num_val_sample = self.batch_size
        hist = model.fit_generator(self.imdb.get_batch(batch_size, phase='train'),
                                   samples_per_epoch=num_train_sample,
                                   nb_epoch=self.nb_epoch,
                                   validation_data=self.imdb.get_batch(batch_size, phase='val'),
                                   nb_val_samples=num_val_sample,
                                   callbacks=[checkpointer])

        self._dump_history(hist.history, True, 'history.log')
        print('Training complete. Saved model as: ', self.model_file)

def set_train_config(common_cfg_file, train_cfg_file, train_mode):
    with open(common_cfg_file) as f: dataset_config = json.load(f)
    with open(train_cfg_file) as f: train_config = json.load(f)

    train_config['train_mode'] = train_mode
    train_config['shear_range'] = np.pi * train_config['shear_range']

    return dataset_config, train_config

def train_net(
        common_cfg_file,
        train_cfg_file,
        train_mode,
        model_file,
        nb_epoch,
        retrain,
        initial_epoch):
    dataset_args, train_args = set_train_config(
        common_cfg_file, train_cfg_file, train_mode)

    # Open and initialize dataset for training
    imdb = Dataset(dataset_args)
    imdb.prep_training(train_args)

    sw = SolverWrapper(
        imdb, model_file, nb_epoch, train_args['batch_size'],
       dataset_args['train_dir'], retrain, initial_epoch)

    sw.train_model()
   