from __future__ import print_function

import os
import cv2
import numpy as np
import random
import glob

from keras.preprocessing.image import transform_matrix_offset_center, apply_transform
from scipy.ndimage.filters import gaussian_filter
from keras import backend as K

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
        
        self.positive_sample = 0 # Match
        self.negative_sample = 1 # Missmatch

        self.remove_outline = args_dict['discard_outline']

        if self.remove_outline:
            if self.wd == 128 and self.ht == 64:
                self.mean_image_name = os.path.join('resources', 'mean_image_128x64.png')
                self.outline_image_name = os.path.join('resources', 'outline_image_128x64.png')
                self.stddev_image_name = os.path.join('resources', 'stddev_image_128x64.png')
            elif self.wd == 256 and self.ht == 128:
                self.mean_image_name = os.path.join('resources', 'mean_image_256x128.png')
                self.outline_image_name = os.path.join('resources', 'outline_image_256x128.png')
                self.stddev_image_name = os.path.join('resources', 'stddev_image_256x128.png')
            elif self.wd == 512 and self.ht == 256:
                self.mean_image_name = os.path.join('resources', 'mean_image_512x256.png')
                self.outline_image_name = os.path.join('resources', 'outline_image_512x256.png')
                self.stddev_image_name = os.path.join('resources', 'stddev_image_512x256.png')
            else:
                print('Image dimension ht:{0:d} wd:{1:d} not supported.'\
                      .format(self.ht, self.wd))
        else:
            self.mean_image_name = ''
            self.stddev_image_name = ''
            self.outline_image_name = ''
            self.mean_image = None
            self.outline_image = None
            self.stddev_image = None

        self.training_ignore_list_file = os.path.join('resources', 'training_ignore_list.txt')
        self.limited_search_space = []

        self._print_dataset_config()
        
    def _print_dataset_config(self):
        return

    def get_input_dim(self):
        if K.image_dim_ordering() == 'tf':
            return (self.ht, self.wd, 1)
        else:
            return (1, self.ht, self.wd)

    def _get_sketch(self, sketch_path):
        sketch = cv2.imread(sketch_path)
        if sketch is not None:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
            sketch = cv2.resize(sketch, (self.wd, self.ht))
            sketch = 255 - sketch # Make background zero by inverting
            sketch = sketch.astype('float32')

            # Some sketches a greyish background. Eliminate it.
            # sketch[sketch < 30] = 0

            if self.remove_outline:
                self.outline_image = self.outline_image.astype('bool')
                sketch = np.ma.filled(np.ma.masked_array(sketch, self.outline_image), 0)
            # Bring the image data to -1 to +1 range
            sketch = (sketch * 2)/255. - 1
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

        if self.remove_outline:
            self.mean_image, self.stddev_image = self.get_mean_sketch()
            self.outline_image = self.get_outline_sketch()

        self.train_pairs, \
        self.train_labels, \
        self.val_pairs, \
        self.val_labels = self._attach_pairs()

        self._print_train_config()
        
    def _print_train_config(self):
        return

    def prep_test(self, test_args):
        if self.remove_outline:
            self.mean_image, self.stddev_image = self.get_mean_sketch()

    def _get_sketch_list(self):
        sketch_list = os.listdir(self.train_dir)
        if len(sketch_list) < 1:
            print('Found only {0:d} sketches in the sketch directory: {1:s}'.\
                format(len(sketch_list), self.train_dir),
                'What are you trying to do? Aborting for now.')
            exit(0)

        # Get list of sketches to ignore
        ignore_list = open(self.training_ignore_list_file, 'r').read().splitlines()
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
                sketch = 255 - sketch  # Make background zero by inverting
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
        
    def get_outline_sketch(self):
        if os.path.exists(self.outline_image_name):
            print('Reading outline image from disk: ', self.outline_image_name)
            outline_matrix = cv2.imread(self.outline_image_name, cv2.IMREAD_GRAYSCALE).astype('float32')
            # Either use IMREAD_GRAYSCALE while reading or convert sketch to Gryyscale after opening

            if outline_matrix is None:
                print('Unable to open {0:s} or {1:s}'.format(self.outline_image_name))
            return outline_matrix

        if not os.path.exists(self.mean_image_name):
            print('Error: Mean image does not exist.')
            return None

        # If we average all images, we get outline
        outline_matrix = cv2.imread(self.mean_image_name, cv2.IMREAD_GRAYSCALE).astype('float32')

        # Smooth the image
        outline_matrix = gaussian_filter(outline_matrix, 1.5, mode='constant', cval=0)
        threshold = 20 # Based on experimentation
        outline_matrix[outline_matrix < threshold] = 0
        outline_matrix[outline_matrix > threshold] = 255

        # write to file for future usage
        print('Writing outline image : {0:s} to disk'.format(self.outline_image_name), end='')
        cv2.imwrite(self.outline_image_name, outline_matrix.astype('uint8'))
        print(' Done')
        return outline_matrix


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
        train_labels = [self.positive_sample, self.negative_sample] * len(train_sketches) * (1 + num_additional)
        for i, sketch_name in enumerate(train_sketches):
            # num_additional adds 2 pairs per sketch
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
        val_labels = [self.positive_sample, self.negative_sample] * len(val_sketches) * (1 + num_additional)
        for i, sketch_name in enumerate(val_sketches):
            # num_additional adds 2 pairs per sketch        
            for _ in range(1 + num_additional):
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

        if K.image_dim_ordering() == 'tf':
            img_shape = (ht, wd, 1)
        else:
            img_shape = (1, ht, wd)
        X_l = np.zeros((batch_size,) + img_shape)
        X_r = np.zeros((batch_size,) + img_shape)
        y = np.array([self.positive_sample, self.negative_sample] * int(batch_size / 2))

        src_idx = 0
        dst_idx = 0
        while True:
            sketch1_name = pairs[src_idx][0]
            sketch2_name = pairs[src_idx + 1][1]
            sketch1 = self._get_sketch(os.path.join(sketch_dir, sketch1_name)).reshape(img_shape)
            sketch2 = self._get_sketch(os.path.join(sketch_dir, sketch2_name)).reshape(img_shape)

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


    def _dump_sketch_pairs(self, sketch_pairs, phase, num_samples_to_dump=None):
        sketch_dir = self.train_dir

        if num_samples_to_dump is None:
            num_pairs = int(len(sketch_pairs) / 2)
        else:
            num_pairs = int(num_samples_to_dump/2)
        labels = np.zeros(num_pairs)  # Won't be used
        batch_size = 2
        wd = self.wd
        ht = self.ht

        my_generator = self.get_batch(batch_size, phase)
        for i in range(0, num_pairs, 2):
            X, y = next(my_generator)
            sketch1 = (1 + X[0][0]) * 255/2.
            sketch2 = (1 + X[0][1]) * 255/2.
            sketch1 = np.clip(sketch1, 0, 255)
            sketch2 = np.clip(sketch2, 0, 255)
            sketch = np.concatenate((sketch1, sketch2), axis=1)
            sketch = sketch.reshape((2 * ht, wd))
            sketch = sketch.astype('uint8')
            cv2.imwrite(os.path.join('temp', phase + '_' + str(i)) + '_0.jpg', sketch)

            sketch1 = (1 + X[1][0]) * 255/2.
            sketch2 = (1 + X[1][1]) * 255/2.
            sketch1 = np.clip(sketch1, 0, 255)
            sketch2 = np.clip(sketch2, 0, 255)
            sketch = np.concatenate((sketch1, sketch2), axis=1)
            sketch = sketch.reshape((2 * ht, wd))
            sketch = sketch.astype('uint8')
            cv2.imwrite(os.path.join('temp', phase + '_' + str(i) + '_1.jpg'), sketch)

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
            sketch = (1 + sketch) * 255/2.
            sketch = np.clip(sketch, 0, 255)
            cv2.imwrite( os.path.join('temp', 'Aug_' + sketch_names[i]), sketch)
            if count >= num_sketches_to_dump:
                break

    def _compare_labels(self, gen, pairs, labels, num_batches, batch_size):
        num_matches = 0
        for batch_id in range(num_batches):
            [x_l, x_r], y = next(gen)
            print('\rbatch_id: {0:d}/{1:d}'.format(batch_id, num_batches), end='')
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
            print('Dumping sketch pairs for debugging....')
            dump_train = dump_val = 500
            self._dump_sketch_pairs(self.train_pairs, 'train', dump_train)
            self._dump_sketch_pairs(self.val_pairs, 'val', dump_val)
            exit(0) # Usually I want to exit after dumping sketches
        if 0:
            print('Dumping sketchs for debugging....')
            num_sketches_to_dump = 10
            self._dump_sketces(num_sketches_to_dump)
            exit(0) # Usually I want to exit after dumping sketches

        if 0:
            print('Testing generators for debugging.....')
            dump_train = dump_val = 10
            self._test_generators(batch_size, dump_train, dump_val)

    '''
    Load sketches from a direcctory. Return the data and IDs.
    Used by the testing module.
    May have to replace it if the test set becomes too large to fit in memory.
    '''
    def load_sketches(self, sketch_dir, limit_search_space):
        if not os.path.exists(sketch_dir):
            print('Error: The sketch directory {0:s} does not exist'.format(sketch_dir))
            return None, None

        sketch_names = os.listdir(sketch_dir)
        random.shuffle(sketch_names) # Shuffle
        #sketch_names = sketch_names[-100:] # Enable for debugging purpose
        
        if len(sketch_names) < 1:
            print('Error: Found only {0:d} sketches in the sketch directory: {1:s}'.\
                format(len(sketch_names), sketch_dir),
                'What are you trying to do? Aborting for now.')
            exit(0)

        # Preparing IDs
        ID = [x.split('.')[0] for x in sketch_names] # sketch names w/o extension
        ID = [x.split('_')[0] for x in ID] # Removing '_' from filenames

        # Limit search space to the correct sketches
        if limit_search_space:
            # Save sketch list from test_set
            if not self.limited_search_space: 
                lss = []
                for i in range(len(sketch_names)):
                    lss.append(ID[i] + '.' + sketch_names[i].split('.')[1])
                self.limited_search_space = lss
                
                # Preparing sketch data
                print('Reading sketches from {0:s}'.format(sketch_dir))
                X = np.empty([len(sketch_names), self.ht, self.wd], dtype='float32')
                for idx, sketch_name in enumerate(sketch_names):
                    print(('\r{0:d}/{1:d} '.format(idx+1, len(sketch_names))), end='')
                    sketch = self._get_sketch(os.path.join(sketch_dir, sketch_name))
                    if sketch is not None:
                        X[idx] = sketch
                print('Done.')                
            else:
                # Uded the sketch list from test_set
                sketch_names = self.limited_search_space
                num_limited_sketches = 0 # In case want to test functionality for debugging
                if num_limited_sketches != 0:
                    sketch_names = sketch_names[:num_limited_sketches]
                ID = [x.split('.')[0] for x in sketch_names]  # sketch names w/o extension
                ID = [x.split('_')[0] for x in ID]  # Removing '_' from filenames
                ID = list(set(ID)) # Removing duplicates since test ID may have >1 sketches per manatee e.g. _A, _B 
                
                # Preparing sketch data
                print('Reading sketches from {0:s}'.format(sketch_dir))
                X = np.empty([len(ID), self.ht, self.wd], dtype='float32')
                for idx, id in enumerate(ID):
                    print(('\r{0:d}/{1:d} '.format(idx+1, len(ID))), end='')
                    # Workaround: Some IDs have different extension in test and train dir
                    sketch_with_path = glob.glob(os.path.join(sketch_dir, id + '.*'))[0]
                    sketch = self._get_sketch(sketch_with_path)
                    if sketch is not None:
                        X[idx] = sketch
                print('Done.')
        else:
            # Preparing sketch data
            print('Reading sketches from {0:s}'.format(sketch_dir))
            X = np.empty([len(sketch_names), self.ht, self.wd], dtype='float32')
            for idx, sketch_name in enumerate(sketch_names):
                print(('\r{0:d}/{1:d} '.format(idx+1, len(sketch_names))), end='')
                sketch = self._get_sketch(os.path.join(sketch_dir, sketch_name))
                if sketch is not None:
                    X[idx] = sketch
            print('Done.')
        
        X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
        return X, ID