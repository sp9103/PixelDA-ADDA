import logging
import numpy as np
import tensorflow as tf

def preprocessing(inputs):
    inputs = tf.cast(inputs, tf.float32)\

    #rgb to gray
    logging.info('Converting RGB images to grayscale')
    inputs = rgb2gray(inputs)

    #Resize
    size = 28
    logging.info('Resizing images to [{}, {}]'.format(size, size))
    inputs = tf.image.resize_images(inputs, [size, size])

    return inputs

RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

def rgb2gray(image):
    return tf.reduce_sum(tf.multiply(image, tf.constant(RGB2GRAY)),
                         3,
                         keep_dims=True)

def gray2rgb(image):
    return tf.multiply(image, tf.constant(RGB2GRAY))
