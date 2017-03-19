from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda, Merge
from keras.optimizers import RMSprop
from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def get_abs_diff(vects):
    x, y = vects
    val = K.abs(x - y)
    return val

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1)

def my_concat(vects):
    x, y = vects
    return K.concatenate([x, y], axis=1)

def my_concat_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 2 * shape1[1])

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

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
    model.add(Dense(1024, activation='sigmoid'))

    input_a = Input(shape=(input_dim))
    input_b = Input(shape=(input_dim))

    # because we re-use the same instance `model`,
    # the weights of the network will be shared across the two branches
    processed_a = model(input_a)
    processed_b = model(input_b)

    if 0:
        # Euclidean distance layer
        temp = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([processed_a, processed_b])
        score = Dense(1, activation = 'sigmoid')(temp) #Dissimilarity score
        model = Model(input=[input_a, input_b], output=score)
    elif 0:
        # Concat Layer
        temp = Lambda(my_concat, output_shape=my_concat_output_shape)([processed_a, processed_b])
        score = Dense(512, activation = 'relu')(temp) #Dissimilarity score
        score = Dense(128, activation = 'relu')(temp)
        score = Dense(1, activation = 'sigmoid')(score)
        model = Model(input=[input_a, input_b], output=score)
    else: # Default: Absolute layer
        temp = Lambda(get_abs_diff, output_shape=abs_diff_output_shape)([processed_a, processed_b])
        temp = Dense(1024, activation='sigmoid')(temp)
        score = Dense(1, activation='sigmoid')(temp)  # Dissimilarity score
        model = Model(input=[input_a, input_b], output=score)

    # Optimizer
    rms = RMSprop(lr=0.001, decay=0.1)

    if 0: # Contrastive loss
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
    else: # Default
        model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

# from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png')

    return model