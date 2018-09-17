import tensorflow as tf
from tensorflow.contrib import slim

from contextlib import ExitStack
from collections import OrderedDict

def LeNet(image,
          is_training = True,
          num_clasess=10,
          reuse_private=False,
          private_scope='mnist',
          reuse_shared=False,
          shared_scope='task_model'):
    layers = OrderedDict()
    net = image

    with ExitStack() as stack:
        stack.enter_context(
            slim.arg_scope(
                [slim.fully_connected, slim.conv2d],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(2.5e-5)))
        stack.enter_context(slim.arg_scope([slim.conv2d], padding='VALID'))

        with tf.variable_scope(private_scope, reuse=reuse_private):
            net = slim.conv2d(net, 20, 5, scope='conv1')
            layers['conv1'] = net
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            layers['pool1'] = net

        with tf.variable_scope(shared_scope, reuse=reuse_shared):
            net = slim.conv2d(net, 50, 5, scope='conv2')
            layers['conv2'] = net
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            layers['pool2'] = net
            net = tf.contrib.layers.flatten(net)
            net = slim.fully_connected(net, 500, scope='fc3')
            layers['fc3'] = net
            net = slim.fully_connected(net, num_clasess, activation_fn=None, scope='fc4')
            layers['fc4'] = net

    return net, layers