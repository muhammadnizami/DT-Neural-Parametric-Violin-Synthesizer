from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K
import numpy as np

class GaussianSpikeNoise(Layer):
    """Apply additive zero-centered Gaussian spike with probability.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the spike distribution.
        p: the probability of the spike appearing
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, stddev, p, **kwargs):
        super(GaussianSpikeNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.p = p

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_binomial(
                K.shape(inputs),
                p=self.p
            ) * K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev,'p': self.p}
        base_config = super(GaussianSpikeNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
