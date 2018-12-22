
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import InputLayer
tfd = tfp.distributions
import config as cf

def bayesian_alexnet(images):

    def _untransformed_scale_constraint(t):
        return tf.clip_by_value(t, -1000, tf.log(cf.kernel_posterior_scale_constraint))
    
    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(untransformed_scale_initializer=tf.random_normal_initializer(
        mean=cf.kernel_posterior_scale_mean,stddev=cf.kernel_posterior_scale_stddev),
        untransformed_scale_constraint=_untransformed_scale_constraint)


    model = tf.keras.Sequential([
    	tf.keras.layers.InputLayer(input_shape=(227,227,3)),
        tfp.layers.Convolution2DFlipout(96,
                                        kernel_size=11,
                                        padding="SAME",
                                        strides = (4,4),
                                        activation=tf.nn.relu,
                                        kernel_posterior_fn = kernel_posterior_fn),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[1, 1],
                                     padding="VALID"),
        
        tfp.layers.Convolution2DFlipout(256,
                                        kernel_size=5,
                                        padding="SAME",
                                        strides = (1,1),
                                        activation=tf.nn.relu,
                                        kernel_posterior_fn = kernel_posterior_fn),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[1, 1],
                                     padding="VALID"),

        tfp.layers.Convolution2DFlipout(384,
                                        kernel_size=3,
                                        padding="SAME",
                                        strides = (1,1),
                                        activation=tf.nn.relu,
                                        kernel_posterior_fn = kernel_posterior_fn),

        tfp.layers.Convolution2DFlipout(384,
                                        kernel_size=3,
                                        padding="SAME",
                                        strides = (1,1),
                                        activation=tf.nn.relu,
                                        kernel_posterior_fn = kernel_posterior_fn),

        tfp.layers.Convolution2DFlipout(256,
                                        kernel_size=3,
                                        padding="SAME",
                                        strides = (1,1),
                                        activation=tf.nn.relu,
                                        kernel_posterior_fn = kernel_posterior_fn),

        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(4096, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(4096, activation = tf.nn.relu),
        tfp.layers.DenseFlipout(cf.num_classes, activation=tf.nn.softmax)])

    logits = model(images)
    targets_distribution = tfd.Categorical(logits=logits)

    return model,logits, targets_distribution
