from __future__ import print_function
import sys #for flushing to stdout
import numpy as np
import cv2
from keras.models import load_model
import json
import csv
import os

from dataset import Dataset
from eval import eval_score_table
from siamese_model import contrastive_loss

class Test():
    def __init__(self, imdb, model_file, train_dir, test_dir):
        self.imdb = imdb
        self.model_file = model_file
        self.input_dim = self.imdb.get_input_dim()
        self.test_dir = test_dir
        self.ranks = sorted([1, 5, 10, 20, 50, 100, 200])
        self.dump_score_table = True # For debugging
        self.limit_search_space = 0 #True
        
        # Use pre-trained model_file
        print('Reading model from disk: ', model_file)
        # self.net = load_model(model_file, custom_objects={'contrastive_loss':contrastive_loss})
        self.net = load_model(model_file)

    def test_single_source(self, model, X, ID):
        print('Computing rank-based accuracy... ')
        num_samples = X.shape[0]
        num_pairs = int((num_samples * (num_samples + 1)) / 2)

        ranks = [rank for rank in self.ranks if rank <= num_samples] # filter bad ranks
        accuracy = [0.] * len(ranks)
        
        size = 1
        for i in [num_pairs, 2] + [i for i in X[0].shape]:
            size *= i
        
        # To tackle memory constraints, process pairs in small sized batches
        batch_size = 128   # Num pairs in a batch
        scores = np.empty(num_pairs).astype('float32')         # List to store scores of ALL pairs
        
        pairs = np.empty([batch_size, 2] + [i for i in X[0].shape]);
        pair_count = 0; counter = 0
        for i in range(num_pairs):
            for j in range(i, num_samples):
                pairs[pair_count] = np.array([[X[i], X[j]]])
                pair_count += 1
                
                if (pair_count == batch_size):
                    print('\r Predicting {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                    batch_scores = model.predict([pairs[:, 0], pairs[:, 1]], batch_size=batch_size)
                    scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                    pair_count = 0
                    counter += batch_size
                    
        # Last remaining pairs which couldn't get processed in main loop
        if pair_count > 0:
            batch_scores = model.predict([pairs[:pair_count, 0], pairs[:pair_count, 1]])
            scores[counter:(counter+pair_count)] = batch_scores[:, 0]
        
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
            
        eval_score_table(score_table, ranks, ID, ID)        


    def _dump_score_table(self, score_table, row_IDs, col_IDs):
        import sys

        if sys.version_info[0] == 2:  # Not named on 2.6
            access = 'wb'
            kwargs = {}
        else:
            access = 'wt'
            kwargs = {'newline':''}    
    
        score_table = np.array(score_table)
        dump = []
        dump += [['', ':'] + col_IDs]
        for row in range(score_table.shape[0]):
            dump += [[row_IDs[row], ':'] + score_table[row].tolist()]
        with open('score_table.csv', access, **kwargs) as f:
            wr = csv.writer(f, delimiter=',')
            for row in dump:
                wr.writerow(row)

        sorted_idx = np.argsort(score_table, axis=1)
        sorted_IDs = []
        sorted_scores = []
        col_IDs_np = np.array(col_IDs)
        for row in range(score_table.shape[0]):
            sorted_IDs += [[row_IDs[row], ':'] + col_IDs_np[sorted_idx[row]].tolist()]
            sorted_scores += [[row_IDs[row], ':'] + score_table[row, sorted_idx[row]].tolist()]

        with open('score_table_sorted_IDs.csv', access, **kwargs) as f:
            wr = csv.writer(f, delimiter=',')
            for row in sorted_IDs:
                wr.writerow(row)

        with open('score_table_sorted_scores.csv', access, **kwargs) as f:
            wr = csv.writer(f, delimiter=',')
            for row in sorted_scores:
                wr.writerow(row)

    def perform_testing(self, model, X1, ID1, X2=None, ID2=None):
        test_single_source = (X2 is None) or (ID2 is None)

        print('Computing rank-based accuracy... ')
        print('Testing single source: ', test_single_source)
        num_rows = X1.shape[0]
        
        if test_single_source:
            X2 = X1; ID2 = ID1
            num_cols = num_rows
            num_pairs = int((num_rows * (num_rows + 1)) / 2)
        else:
            num_cols = X2.shape[0]
            num_pairs = num_rows * num_cols
        
        ranks = [rank for rank in self.ranks if rank <= num_cols] # filter bad ranks
        
        # To tackle memory constraints, process pairs in small sized batchs
        batch_size = 32    # Num pairs in a batch
        scores = np.empty(num_pairs).astype('float32') # List to store scores of ALL pairs     
        score_table = np.zeros([num_rows, num_cols]).astype('float32')
        
        if test_single_source:
            '''
            Compares sketches from a single diectory to the sketches in the same directory.
            We use this knowledge to avoid computation of permutations of a given pair. That 
            is, each possible combination only passes once through network.
            '''        
            batch = np.empty([batch_size, 2] + [i for i in X1[0].shape])
            pair_count = 0; counter = 0
            for i in range(num_rows):
                for j in range(i, num_cols):
                    batch[pair_count] = np.array([[X1[i], X1[j]]])
                    pair_count += 1
                    
                    if (pair_count == batch_size):
                        print('\r Predicting pair {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                        batch_scores = model.predict([batch[:, 0], batch[:, 1]], batch_size=batch_size)
                        scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                        pair_count = 0
                        counter += batch_size
                        
            # Last remaining pairs which couldn't get processed in main loop
            if pair_count > 0:
                print('\r Predicting pair {0:d}/{1:d}'.format(num_pairs, num_pairs), end=''); sys.stdout.flush()
                batch_scores = model.predict([batch[:pair_count, 0], batch[:pair_count, 1]])
                scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                
            print('Analyzing scores... ', end='')
            idx = 0
            for i in range(num_rows):
                for j in range(i, num_cols):
                    score_table[i, j] = scores[idx]
                    score_table[j, i] = scores[idx]
                    idx = idx + 1
        else:
            '''
            No shortcuts about permutations are made. All sketches in X1 are compared with
            all sketches in X2.
            '''
            batch = np.empty([batch_size, 2] + [i for i in X1[0].shape])
            pair_count = 0; counter = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    batch[pair_count] = np.array([[X1[i], X2[j]]])
                    pair_count += 1
                    
                    if (pair_count == batch_size):
                        print('\r Predicting pair {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                        batch_scores = model.predict([batch[:, 0], batch[:, 1]], batch_size=batch_size)
                        scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                        pair_count = 0
                        counter += batch_size
                        
            # Last remaining pairs which couldn't get processed in main loop
            if pair_count > 0:
                print('\r Predicting pair {0:d}/{1:d}'.format(num_pairs, num_pairs), end=''); sys.stdout.flush()
                batch_scores = model.predict([batch[:pair_count, 0], batch[:pair_count, 1]])
                scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                
            print('Analyzing scores... ', end='')
            idx = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    score_table[i, j] = scores[idx]
                    idx = idx + 1         
        
        # Parse score table and generate accuracy metrics
        eval_score_table(score_table, ranks, ID1, ID2)        
        
        if self.dump_score_table:
            self._dump_score_table(score_table, ID1, ID2)
        

def debug_sketches(X1, X2, ht, wd) :
    for i in range(X1.shape[0]):
        print(np.min(X1[i]), np.mean(X1[i]), np.max(X1[i]))
    
    for i in range(X1.shape[0]):
        sketch = X1[i].reshape(ht, wd)
        sketch = sketch - np.min(sketch)
        sketch = sketch * 255/np.max(sketch)
        sketch = sketch.astype('uint8')
        file_name = 'test_' + str(i) + '_' + str(wd) + 'x' + str(ht)+ '.png'
        cv2.imwrite(file_name, sketch)


def dump_train_test_sketch_pairs(X1, ID1, X2, ID2):
    for i, id in enumerate(ID1):
        stripped_id = id.split('.')[0].split('_')[0]
        if stripped_id in ID2:
            sketch1 = X1[i]
            sketch2 = X2[ID2.index(stripped_id)]
            sketch1 = (1 + sketch1) * 255/2.
            sketch2 = (1 + sketch2) * 255/2.
            sketch1 = 255 - np.clip(sketch1, 0, 255)
            sketch2 = 255 - np.clip(sketch2, 0, 255)
            sketch = np.concatenate((sketch2, sketch1), axis=1) # Place Train-sketch on top of test-sketch
            # sketch.shape is (1, ht, wd) at this point
            sketch = sketch.reshape(sketch.shape[1], sketch.shape[1]) # make shape (ht, wd)
            sketch = sketch.astype('uint8')
            cv2.imwrite(os.path.join('temp', id + '_pair.jpg'), sketch)
        else:
            print("Warning: Stripped ID: {0:s} not found in test set. Skipping pair dump.".format(stripped_id))
    exit(0)

def set_test_config(common_cfg_file, test_cfg_file):
    with open(common_cfg_file) as f: dataset_config = json.load(f)
    with open(test_cfg_file) as f: test_config = json.load(f)

    return dataset_config, test_config

def test_net(common_cfg_file, test_cfg_file, test_mode, model_file):
    dataset_args, test_args = set_test_config(common_cfg_file, test_cfg_file)

    imdb = Dataset(dataset_args)
    imdb.prep_test(test_args)

    sw = Test(imdb, model_file, dataset_args['train_dir'], dataset_args['test_dir'])

    if test_mode == 0:
        # searching for test_dir sketches inside test_dir
        X, ID = sw.imdb.load_sketches(dataset_args['test_dir'], sw.limit_search_space)
        # sw.test_single_source(sw.net, X, ID)
        sw.perform_testing(sw.net, X, ID)
    elif test_mode == 1:
        # searching for test_dir sketches inside train_dir
        X1, ID1 = sw.imdb.load_sketches(dataset_args['test_dir'], sw.limit_search_space)
        X2, ID2 = sw.imdb.load_sketches(dataset_args['train_dir'], sw.limit_search_space)
        # dump_train_test_sketch_pairs(X1, ID1, X2, ID2)
        sw.perform_testing(sw.net, X1, ID1, X2, ID2)
    return
