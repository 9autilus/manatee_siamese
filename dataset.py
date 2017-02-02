from __future__ import print_function

import os
import cv2
import numpy as np
import random

class dataset():
    def __init__(self, train_dir=None, test_dir=None, test2_dir=None):
        self.wd = 128#360
        self.ht = 64#180
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test2_dir = test2_dir

    def get_input_dim(self):
        return (1, self.ht, self.wd)

    def get_sketch(self, sketch_path):
        sketch = cv2.imread(sketch_path)
        if sketch is not None:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
            sketch = cv2.resize(sketch, (self.wd, self.ht))
            #inverting sketch for a black background
            sketch = (255 - sketch).astype('float32') 
            # Zero mean and Unit variance
            sketch = (sketch - sketch.mean())/sketch.var()
            return sketch
        else:
            print('Unable to open ', sketch_path, ' Skipping.')
            return None
        
    def load_sketches(self, sketch_dir):
        if not os.path.exists(sketch_dir):
            print('The sketch directory {0:s} does not exist'.format(sketch_dir))
            return None, None

        sketch_names = os.listdir(sketch_dir)
        random.shuffle(sketch_names) # Shuffle
        sketch_names = sketch_names[-100:] # Enable for debugging purpose
        
        if len(sketch_names) < 1:
            print('Found only {0:d} sketches in the sketch directory: {1:s}'.\
                format(len(sketch_names), sketch_dir),
                'What are you trying to do? Aborting for now.')
            exit(0)    
            
        print('Reading sketches from {0:s}'.format(sketch_dir))
        ID = [x.split('.')[0] for x in sketch_names] # sketch names w/o extension
        X = np.empty([len(sketch_names), self.ht, self.wd], dtype='float32')
        for idx, sketch_name in enumerate(sketch_names):
            print(('\r{0:d}/{1:d} '.format(idx+1, len(sketch_names))), end='')
            sketch = self.get_sketch(os.path.join(sketch_dir, sketch_name))
            if sketch is not None:
                X[idx] = sketch
        print('Done.')
        
        X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
        return X, ID