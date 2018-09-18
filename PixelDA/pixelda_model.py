import numpy as np
import functools
import math

import tensorflow as tf
from common import classifier

slim = tf.contrib.slim

def create_model(target_images,
                 source_images,
                 source_labels,
                 num_classes,
                 is_training=False):
    # generator
    generator = resnet_generator(source_images, target_images.shape.as_list()[1:4])

    #discriminator -> reuse하는 두개를 만듦. 이미지 사이즈는 커서.. 두개 겹쳐서 사용하기에는 메모리 문제가 있을지도
    discriminator = dict()
    discriminator['transferred_domain_logits'] = predict_domain(
        generator,
        is_training,
        False)
    discriminator['target_domain_logits'] = predict_domain(
        target_images,
        is_training,
        True)

    #classifier

    with tf.variable_scope('classifier'):
        classifierDict = dict()
        classifierDict['source_task_logits'] = classifier.LeNet(source_images,
                                                                False,
                                                                num_classes,
                                                                reuse_private=False,
                                                                private_scope='source_task_classifier',
                                                                reuse_shared=False)
        classifierDict['transferred_task_logits'] = classifier.LeNet(source_images,
                                                                     False,
                                                                     num_classes,
                                                                     reuse_private=False,
                                                                     private_scope='transferred_task_classifier',
                                                                     reuse_shared=True)
        classifierDict['target_task_logits'] = classifier.LeNet(source_images,
                                                                False,
                                                                num_classes,
                                                                reuse_private=True,
                                                                private_scope='transferred_task_classifier',
                                                                reuse_shared=True)
    return generator, discriminator, classifierDict

def resnet_generator(images, output_shape):
    with tf.variable_scope('generator'):
        noise = noise_layer(10, images.shape.as_list())
        images = tf.concat([images, noise], 3)

        net = resnet_stack(images, output_shape, 'resnet_stack')
    return net

## value를 지정해주지 않음. 문제가 생길지는 확인해봐야함
def resnet_stack(images, output_shape, scope = None):
    with tf.variable_scope(scope, 'resnet_style_transfer'):
        with slim.arg_scope(
                [slim.conv2d],
                normalizer_fn=slim.batch_norm,
                kernel_size=[3] * 2,
                stride=1):
            net = slim.conv2d(
                images,
                64,
                normalizer_fn=None,
                activation_fn=tf.nn.relu)

            for i in range(6):
                net = resnet_block(net)

            net = slim.conv2d(
                net,
                output_shape[-1],
                kernel_size=[1, 1],
                normalizer_fn=None,
                activation_fn=tf.nn.tanh,
                scope='conv_out')
    return net

def resnet_block(net):
    net_in = net
    net = slim.conv2d(
        net,
        64,
        stride=1,
        normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu)
    net = slim.conv2d(
        net,
        64,
        stride=1,
        normalizer_fn=slim.batch_norm,
        activation_fn=None)
    net += net_in
    return net

def noise_layer(len, img_shape):
    with tf.variable_scope('noise'):
        noise = tf.random_uniform(
            shape=[img_shape[0], len],
            minval=-1,
            maxval=1,
            dtype=tf.float32,
            name='random_noise')
        layer = slim.fully_connected(
            noise, np.asscalar(np.prod(img_shape[1:3])), activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        layer = tf.reshape(layer, img_shape[0:3] + [1])

    tf.logging.info('noise layer size %s volume', layer.shape)
    return layer

def lrelu(x, leakiness=0.2):
  """Relu, with optional leaky support."""
  return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def add_noise(hidden, is_training):
    hidden = slim.dropout(
        hidden,
        0.9,
        is_training=is_training,
        scope='dropout')
    return hidden + tf.random_normal(
        hidden.shape.as_list(),
        mean=0.0,
        stddev=0.2)

def predict_domain(images, is_training=False, reuse=False, scope='discriminator'):
    first_stride = 1
    with tf.variable_scope(scope, 'discriminator', [images], reuse=reuse):
        lrelu_partial = functools.partial(lrelu, leakiness=0.2)  # leaky relu 정의
        with slim.arg_scope(
                [slim.conv2d],
                kernel_size=[3] * 2,
                activation_fn=lrelu_partial,
                stride=2,
                normalizer_fn=slim.batch_norm):
            net = slim.conv2d(
                images,
                64,
                normalizer_fn=None,
                stride=first_stride,
                scope='conv1_stride%s' % first_stride)
            net = add_noise(net, is_training)

            block_id = 2
            while net.shape.as_list()[1] > 4:
                num_filters = 64 * int(math.pow(2, block_id-1))
                net = slim.conv2d(
                    net,
                    num_filters,
                    stride=1,
                    scope='conv_%s' % (block_id))

                net = slim.conv2d(
                    net, num_filters, scope='conv_%s_stride2' % block_id)
                net = add_noise(net, is_training)
                block_id += 1

            net = slim.flatten(net)
            net = slim.fully_connected(
                net,
                1,
                normalizer_fn=None,
                activation_fn=None,
                scope='fc_logit_out')
    return net