from __future__ import print_function

import os
import cv2
import numpy as np
import random

from keras.preprocessing.image import transform_matrix_offset_center, apply_transform

class Dataset():
    def __init__(self, args_dict):
        self.wd = args_dict['wd']
        self.ht = args_dict['ht']
        self.train_dir = args_dict['train_dir']
        self.test_dir = args_dict['test_dir']

        self.train_pairs = None
        self.train_labels = None
        self.val_pairs = None
        self.val_labels = None

        if self.wd == 128 and self.ht == 64:
            self.mean_image_name = os.path.join('resources', 'mean_image_128x64.png')
            self.stddev_image_name = os.path.join('resources', 'stddev_image_128x64.png')
        elif self.wd == 256 and self.ht == 128:
            self.mean_image_name = os.path.join('resources', 'mean_image_256x128.png')
            self.stddev_image_name = os.path.join('resources', 'stddev_image_256x128.png')
        elif self.wd == 512 and self.ht == 256:
            self.mean_image_name = os.path.join('resources', 'mean_image_512x256.png')
            self.stddev_image_name = os.path.join('resources', 'stddev_image_512x256.png')
        else:
            print('Image dimension ht:{0:d} wd:{1:d} not supported.'\
                  .format(self.ht, self.wd))

        self.mean_image, self.stddev_image = self.get_mean_sketch()

        self.ignore_list_file = os.path.join('resources', 'ignore_list.txt')

    def get_input_dim(self):
        return (1, self.ht, self.wd)

    def _get_sketch(self, sketch_path):
        sketch = cv2.imread(sketch_path)
        if sketch is not None:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
            sketch = cv2.resize(sketch, (self.wd, self.ht))
            # Zero mean and Unit variance
            sketch = (sketch.astype('float32') - self.mean_image)/self.stddev_image
            return sketch
        else:
            print('Unable to open ', sketch_path, ' Skipping.')
            return None

    def prep_training(self, train_args):
        self.val_split = train_args['val_split']
        self.use_augmentation = train_args['use_augmentation']
        self.num_additional_sketches = train_args['num_additional_sketches']
        self.height_shift_range = train_args['height_shift_range']
        self.width_shift_range = train_args['width_shift_range']
        self.rotation_range = train_args['rotation_range']
        self.shear_range = train_args['shear_range']
        self.zoom_range = train_args['zoom_range']
        self.fill_mode = train_args['fill_mode']
        self.cval = train_args['cval']

        if not self.use_augmentation:
            self.num_additional_sketches = 0

        self.sketch_list = self._get_sketch_list()

        self.train_pairs, \
        self.train_labels, \
        self.val_pairs, \
        self.val_labels = self._attach_pairs()

        return

    def _get_sketch_list(self):
        sketch_list = os.listdir(self.train_dir)
        if len(sketch_list) < 1:
            print('Found only {0:d} sketches in the sketch directory: {1:s}'.\
                format(len(sketch_list), self.train_dir),
                'What are you trying to do? Aborting for now.')
            exit(0)

        # Get list of sketches to ignore
        ignore_list = open(self.ignore_list_file, 'r').read().splitlines()
        ignore_list = [i for i in ignore_list if i.isspace() is False and i.startswith('#') is False]
        # Remove the sketches that are present in ignore_list
        sketch_list = [i for i in sketch_list if i not in ignore_list]
        # Shuffle the list
        random.shuffle(sketch_list)
        return sketch_list

    '''
    '''
    def get_mean_sketch(self):
        if os.path.exists(self.mean_image_name) and os.path.exists(self.stddev_image_name):
            print('Reading mean image from disk: ', self.mean_image_name, self.stddev_image_name)
            mean_matrix = cv2.imread(self.mean_image_name, cv2.IMREAD_GRAYSCALE).astype('float32')
            stddev_matrix = cv2.imread(self.stddev_image_name, cv2.IMREAD_GRAYSCALE).astype('float32')
            # Either use IMREAD_GRAYSCALE while reading or convert sketch to Gryyscale after opening
            
            if (mean_matrix is None) or (stddev_matrix is None):
                print('Unable to open {0:s} or {1:s}'.format(self.mean_image_name, self.stddev_image_name))
            return mean_matrix, stddev_matrix
        
        # Read all sketches and compute the mean and stddev matrices
        print('Reading sketches from {0:s} to compute mean images:'.format(self.train_dir))

        X = np.empty([len(self.sketch_list), self.ht, self.wd], dtype='float32')
        idx = 0
        for sketch_name in self.sketch_list:
            print(('\r{0:d}/{1:d} '.format(idx+1, len(self.sketch_list))), end='')
            
            sketch_path = os.path.join(self.train_dir, sketch_name)
            sketch = cv2.imread(sketch_path)
            if sketch is not None:
                sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
                sketch = cv2.resize(sketch, (self.wd, self.ht))
                X[idx] = sketch
            else:
                print('Unable to open ', sketch_path, 
                ' Skipping in mean calculation. Mean image not reliable')
            idx += 1
        print('Done.')
        
        mean_matrix = np.mean(X, axis=0).astype('float32')
        stddev_matrix = X.std(axis=0).astype('float32')
        
        # write to file for future usage
        print('Writing mean images : {0:s} {1:s} to disk'.format(self.mean_image_name, self.stddev_image_name), end='')
        cv2.imwrite(self.mean_image_name, mean_matrix.astype('uint8'))
        cv2.imwrite(self.stddev_image_name, stddev_matrix.astype('uint8'))
        print(' Done')
        
        return mean_matrix, stddev_matrix
        
            
    def get_num_train_sample(self):
        return len(self.train_pairs)

    def get_num_val_sample(self):
        return len(self.val_pairs)

    def _attach_pairs(self):
        sketch_dir = self.train_dir
        num_additional = self.num_additional_sketches
        sketch_names = self.sketch_list
        # sketch_names = sketch_names[-100:] # Enable for debugging purpose
        train_split = 100 - self.val_split
        # Divide between training and validation set
        num_sketches_train = int((len(sketch_names) * train_split) / 100)

        train_sketches = sketch_names[:num_sketches_train]
        val_sketches = sketch_names[num_sketches_train:]

        # Create training pairs
        train_pairs = []
        train_labels = [0, 1] * len(train_sketches) * (1 + num_additional)
        for i, sketch_name in enumerate(train_sketches):
            for _ in range(1 + num_additional):
                # Positive pair
                train_pairs.append([sketch_name, sketch_name])
                # Negative pair
                rand_idx = i
                while rand_idx == i:
                    rand_idx = random.randrange(0, len(train_sketches))
                train_pairs.append([sketch_name, train_sketches[rand_idx]])

        # Create validation pairs
        val_pairs = []
        val_labels = [0, 1] * len(val_sketches) * (1 + num_additional)
        for sketch_name in val_sketches:
            for j in range(1 + num_additional):
                # Positive pair
                val_pairs.append([sketch_name, sketch_name])
                # Negative pair
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
            sketch1 = self._get_sketch(os.path.join(sketch_dir, sketch1_name)).reshape(1, ht, wd)
            sketch2 = self._get_sketch(os.path.join(sketch_dir, sketch2_name)).reshape(1, ht, wd)

            if self.use_augmentation is True:
                # Positive pair
                X_l[dst_idx] = self._apply_affine_distortion(sketch1)
                X_r[dst_idx] = self._apply_affine_distortion(sketch1)
                # Negative pair
                X_l[dst_idx + 1] = self._apply_affine_distortion(sketch1)
                X_r[dst_idx + 1] = self._apply_affine_distortion(sketch2)
            else:
                # Positive pair
                X_l[dst_idx] = sketch1
                X_r[dst_idx] = sketch1
                # Negative pair
                X_l[dst_idx + 1] = sketch1
                X_r[dst_idx + 1] = sketch2

            src_idx += 2
            dst_idx += 2

            if src_idx >= len(pairs):
                src_idx = 0

            if dst_idx >= batch_size:
                dst_idx = 0
                yield [X_l, X_r], y

    # Function definition taken from keras source code
    def _apply_affine_distortion(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 0

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix),
                                  zoom_matrix)

        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=self.fill_mode, cval=self.cval)

        return x


    def _dump_sketch_pairs(self, sketch_pairs, phase, num_samples_to_dump):
        sketch_dir = self.train_dir

        num_pairs = int(num_samples_to_dump/2)
        labels = np.zeros(num_pairs)  # Won't be used
        batch_size = 2
        wd = self.wd
        ht = self.ht

        print(sketch_pairs)
        my_generator = self.get_batch(batch_size, phase)
        for i in range(0, num_pairs, 2):
            X, y = next(my_generator)
            sketch1 = X[0][0]
            sketch2 = X[0][1]
            if self.use_augmentation is True:
                # Scale the sketches to 0-255
                sketch1 = sketch1 - np.min(sketch1)
                sketch1 = sketch1 * (255. / np.max(sketch1))
                sketch2 = sketch2 - np.min(sketch2)
                sketch2 = sketch2 * (255. / np.max(sketch2))
            else:
                sketch1 = (sketch1 * self.stddev_image) + self.mean_image
                sketch2 = (sketch2 * self.stddev_image) + self.mean_image
            sketch = np.concatenate((sketch1, sketch2), axis=1)
            sketch = sketch.reshape((2 * ht, wd))
            sketch = sketch.astype('uint8')
            cv2.imwrite(phase + '_' + str(i) + '_0.jpg', sketch)

            sketch1 = X[1][0]
            sketch2 = X[1][1]
            if self.use_augmentation is True:
                # Scale the sketches to 0-255
                sketch1 = sketch1 - np.min(sketch1)
                sketch1 = sketch1 * (255. / np.max(sketch1))
                sketch2 = sketch2 - np.min(sketch2)
                sketch2 = sketch2 * (255. / np.max(sketch2))
            else:
                sketch1 = (sketch1 * self.stddev_image) + self.mean_image
                sketch2 = (sketch2 * self.stddev_image) + self.mean_image

            sketch = np.concatenate((sketch1, sketch2), axis=1)
            sketch = sketch.reshape((2 * ht, wd))
            sketch = sketch.astype('uint8')
            cv2.imwrite(phase + '_' + str(i) + '_1.jpg', sketch)

    '''
    Function to visualize how the sketches look after making them
    zero mean and unit variance
    '''
    def _dump_sketces(self, num_sketches_to_dump):
        sketch_dir = self.train_dir

        if not os.path.exists(sketch_dir):
            print('The sketch directory {0:s} does not exist'.format(sketch_dir))
            return

        sketch_names = self.sketch_list
        random.shuffle(sketch_names)  # Shuffle
        wd = self.wd
        ht = self.ht

        count = 0
        for i in range(num_sketches_to_dump):
            sketch = self._get_sketch(os.path.join(sketch_dir, sketch_names[i]))
            # This sketch is zero-mean and unit-variance. In order to write
            # it on disk, we first need to bring it 0-255 range
            sketch = sketch - np.min(sketch)
            sketch = sketch * (255. / np.max(sketch))
            cv2.imwrite( 'Aug_' + sketch_names[i], sketch)
            if count >= num_sketches_to_dump:
                break

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
        gen = self.get_batch(batch_size, 'train')
        self._compare_labels(gen, self.train_pairs, self.train_labels, num_train_batches, batch_size)


        print('Testing validation-set labels:')
        gen = self.get_batch(batch_size, 'val')
        self._compare_labels(gen, self.val_pairs, self.val_labels, num_val_batches, batch_size)

    def validate_dataset(self, batch_size):
        ## Dump sketch pairs to file as images stacked on top of one another
        if 0:
            dump_train = dump_val = 50
            self._dump_sketch_pairs(self.train_pairs, 'train', dump_train)
            self._dump_sketch_pairs(self.val_pairs, 'val', dump_val)
            exit(0) # Usually I want to exit after dumping sketches
        if 0:
            num_sketches_to_dump = 100
            self._dump_sketces(num_sketches_to_dump)
            exit(0) # Usually I want to exit after dumping sketches

        if 0:
            dump_train = dump_val = 10
            self._test_generators(batch_size, dump_train, dump_val)

    '''
    Load sketches from a direcctory. Return the data and IDs.
    Used by the testing module.
    May have to replace it if the test set becomes too large to fit in memory.
    '''
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
        
      