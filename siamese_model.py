from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda, Merge
from keras.optimizers import RMSprop
from keras import backend as K

def get_abs_diff(vects):
    x, y = vects
    val = K.abs(x - y)
    return val

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1)

def my_concat(vects):
    x, y = vects
    c = K.concatenate([x, y], axis=1)
    return c

def my_concat_output_shape(shapes):
    shape1, shape2 = shapes
    new_shape = (shape1[0], 2 * shape1[1])
    return new_shape

def create_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid', input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))    
    model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))     
    model.add(Convolution2D(256, 5, 5, activation='relu', border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    input_a = Input(shape=(input_dim))
    input_b = Input(shape=(input_dim))

    # because we re-use the same instance `model`,
    # the weights of the network will be shared across the two branches
    processed_a = model(input_a)
    processed_b = model(input_b)

    if 1:
        # Concat Layer
        temp = Lambda(my_concat, output_shape=my_concat_output_shape)([processed_a, processed_b])
        score = Dense(1, activation = 'sigmoid')(temp) #Dissimilarity score
        model = Model(input=[input_a, input_b], output=score)
    else:
        # Absolute layer
        temp = Lambda(get_abs_diff, output_shape = abs_diff_output_shape)([processed_a, processed_b])
        score = Dense(1, activation = 'sigmoid')(temp) #Dissimilarity score
        model = Model(input=[input_a, input_b], output=score)

    # Optimizer
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png')

    return model