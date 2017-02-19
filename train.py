from __future__ import print_function
from siamese_model import create_network
from dataset import Dataset
from keras.callbacks import ModelCheckpoint
import numpy as np
import random


class SolverWrapper():
    def __init__(self, imdb, weights, nb_epoch, batch_size, train_dir):
        self.imdb = imdb
        self.weights_file = weights
        self.nb_epoch = nb_epoch
        self.input_dim = self.imdb.get_input_dim()
        self.train_dir = train_dir
        self.batch_size = batch_size
        
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

        # self.imdb.validate_dataset(self.batch_size, num_train_sample, num_val_sample)

        # network definition
        input_dim = (1, self.imdb.ht, self.imdb.wd)
        model = create_network(input_dim)

        # Create check point callback
        checkpointer = ModelCheckpoint(filepath=self.weights_file,
                                       monitor='val_loss', verbose=1, save_best_only=True)

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
        print('Training complete. Saved model as: ', self.weights_file)


def train_net(sketch_dir, weights, nb_epoch):
    imdb = Dataset(train_dir=sketch_dir)
    batch_size = 32

    sw = SolverWrapper(imdb, weights, nb_epoch, batch_size, sketch_dir)

    sw.train_model()
   