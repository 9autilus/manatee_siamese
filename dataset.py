from __future__ import print_function

import os
import cv2
import numpy as np
import random

class Dataset():
    def __init__(self, train_dir=None, test_dir=None, test2_dir=None):
        self.wd = 256
        self.ht = 28
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test2_dir = test2_dir

        self.train_pairs, self.train_labels, self.val_pairs, self.val_labels = self._attach_pairs(train_dir)

    def get_input_dim(self):
        return (1, self.ht, self.wd)

    def _get_sketch(self, sketch_path):
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

    def get_num_train_sample(self):
        return 2 * len(self.train_pairs)

    def get_num_val_sample(self):
        return 2 * len(self.val_pairs)

    def _attach_pairs(self, sketch_dir):
        if not os.path.exists(sketch_dir):
            print('The sketch directory {0:s} does not exist'.format(sketch_dir))
            return None, None, None, None

        sketch_names = os.listdir(sketch_dir)
        random.shuffle(sketch_names)  # Shuffle

        # sketch_names = sketch_names[-100:] # Enable for debugging purpose

        # Divide between training and validation set
        num_sketches_train = int((len(sketch_names) * 70) / 100)

        train_sketches = sketch_names[:num_sketches_train]
        val_sketches = sketch_names[num_sketches_train:]

        # Create training pairs
        train_pairs = []
        train_labels = [0, 1] * len(train_sketches)
        for i, sketch_name in enumerate(train_sketches):
            train_pairs.append([sketch_name, sketch_name])

            rand_idx = i
            while rand_idx == i:
                rand_idx = random.randrange(0, len(train_sketches))
            train_pairs.append([sketch_name, train_sketches[rand_idx]])

        # Create validation pairs
        val_pairs = []
        val_labels = [0, 1] * len(val_sketches)
        for sketch_name in val_sketches:
            val_pairs.append([sketch_name, sketch_name])

            rand_idx = i
            while rand_idx == i:
                rand_idx = random.randrange(0, len(val_sketches))

            val_pairs.append([sketch_name, val_sketches[rand_idx]])

        return train_pairs, train_labels, val_pairs, val_labels

    def get_batch(self, batch_size, phase):
        # batch size must be even number
        if batch_size % 2 != 0:
            print('Error: batch size must be an even number')
            exit(0)

        if phase == 'train':
            sketch_dir = self.train_dir
            pairs = self.train_pairs
            labels = self.train_labels
        elif phase == 'val':
            sketch_dir = self.train_dir
            pairs = self.val_pairs
            labels = self.val_labels
        else:
            print('Error: get_batch() received weird "phase":{0:s}'.format(phase))
            exit(0)

        ht = self.ht
        wd = self.wd

        X_l = np.zeros((batch_size, 1, ht, wd))
        X_r = np.zeros((batch_size, 1, ht, wd))
        y = np.array([0, 1] * int(batch_size / 2))

        src_idx = 0
        dst_idx = 0
        while True:
            sketch1_name = pairs[src_idx][0]
            sketch2_name = pairs[src_idx + 1][1]
            sketch1 = self._get_sketch(os.path.join(sketch_dir, sketch1_name))
            sketch2 = self._get_sketch(os.path.join(sketch_dir, sketch2_name))

            X_l[dst_idx] = sketch1.reshape(1, ht, wd)
            X_r[dst_idx] = sketch1.reshape(1, ht, wd)
            X_l[dst_idx + 1] = sketch1.reshape(1, ht, wd)
            X_r[dst_idx + 1] = sketch2.reshape(1, ht, wd)

            src_idx += 2
            dst_idx += 2

            if src_idx >= len(pairs):
                src_idx = 0

            if dst_idx >= batch_size:
                dst_idx = 0
                yield [X_l, X_r], y

    def _dump_sketch_pairs(self, sketch_pairs, prefix):
        sketch_dir = self.train_dir

        num_pairs = len(sketch_pairs)
        labels = np.zeros(num_pairs)  # Won't be used
        batch_size = 2
        wd = self.wd
        ht = self.ht

        print(sketch_pairs)
        my_generator = self.get_train_batch(batch_size)

        for i in range(0, num_pairs, 2):
            X, y = next(my_generator)

            sketch = np.concatenate((X[0][0], X[0][1]), axis=1)
            sketch = sketch.reshape((2 * ht, wd))
            sketch = sketch - np.min(sketch)  # bring lower limit to 0
            sketch = sketch * (255. / np.max(sketch))
            sketch = sketch.astype('uint8')
            cv2.imwrite(prefix + str(i) + '_0.jpg', sketch)

            sketch = np.concatenate((X[1][0], X[1][1]), axis=1)
            sketch = sketch.reshape((2 * ht, wd))
            sketch = sketch - np.min(sketch)  # bring lower limit to 0
            sketch = sketch * (255. / np.max(sketch))
            sketch = sketch.astype('uint8')
            cv2.imwrite(prefix + str(i) + '_1.jpg', sketch)

    def _compare_labels(self, gen, pairs, labels, num_batches, batch_size):
        num_matches = 0
        for batch_id in range(num_batches):
            [x_l, x_r], y = next(gen);
            print('\rbatch_id: {0:d}/{1:d}'.format(batch_id, num_batches), end='');
            for i in range(batch_size):
                num_matches += ((labels[i] == 0) == np.array_equal(x_l[i], x_r[i]))
        print('\nLabels match: {0:.1f}%'.format(100 * num_matches / float(num_batches * batch_size)))

    def _test_generators(self, batch_size, num_train, num_val):
        sketch_dir = self.train_dir

        num_train_batches = np.ceil(num_train / float(batch_size)).astype('int32')
        num_val_batches = np.ceil(num_train / float(batch_size)).astype('int32')

        print('Testing training-set labels:')
        gen = self.get_train_batch(batch_size)
        self._compare_labels(gen, self.train_pairs, self.train_labels, num_train_batches, batch_size)


        print('Testing validation-set labels:')
        gen = self.get_val_batch(batch_size)
        self._compare_labels(gen, self.val_pairs, self.val_labels, num_val_batches, batch_size)

    def validate_dataset(self, batch_size, samples_train_set, samples_val_set):
        ## Dump sketch pairs to file as images stacked on top of one another
        if 0:
            self._dump_sketch_pairs(self.train_pairs, 'tr_')
            self._dump_sketch_pairs(self.val_pairs, 'val_')

        self._test_generators(batch_size, samples_train_set, samples_val_set)

    def load_sketches(self, sketch_dir):
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
        X = np.empty([len(sketch_names), self.ht, self.wd], dtype='float32')
        for idx, sketch_name in enumerate(sketch_names):
            print(('\r{0:d}/{1:d} '.format(idx+1, len(sketch_names))), end='')
            sketch = self._get_sketch(os.path.join(sketch_dir, sketch_name))
            if sketch is not None:
                X[idx] = sketch
        print('Done.')
        
        X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
        return X, ID