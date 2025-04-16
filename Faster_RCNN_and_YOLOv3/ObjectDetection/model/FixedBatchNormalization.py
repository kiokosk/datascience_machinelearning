# from keras.engine import Layer, InputSpec
from keras.layers import Layer, InputSpec

import tensorflow as tf

from keras import initializers, regularizers
from keras import backend as K


class FixedBatchNormalization(Layer):

    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape=shape,
                                            initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape=shape,
                                           initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True
def call(self, x, mask=None):
    assert self.built, 'Layer must be built before being called'
    input_shape = K.int_shape(x)

    # Standard case for axis=-1 (channels last)
    if self.axis == -1 or self.axis == len(input_shape) - 1:
        x_normed = K.batch_normalization(
            x,
            self.running_mean,
            self.running_std,
            self.beta,
            self.gamma,
            epsilon=self.epsilon
        )
    else:
        # Manually normalize for other axes (e.g., channels first)
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        mean = K.reshape(self.running_mean, broadcast_shape)
        std = K.reshape(self.running_std, broadcast_shape)
        beta = K.reshape(self.beta, broadcast_shape)
        gamma = K.reshape(self.gamma, broadcast_shape)

        x_normed = (x - mean) / (std + self.epsilon)
        x_normed = x_normed * gamma + beta

    return x_normed


    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(FixedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
