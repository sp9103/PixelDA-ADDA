import logging
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common import util, classifier
from data_factory import dataset_factory

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('source_dataset', 'mnist', 'The name of the source dataset.'
                    ' If hparams="arch=dcgan", this flag is ignored.')

flags.DEFINE_string('target_dataset', 'mnist_m',
                    'The name of the target dataset.')

flags.DEFINE_string('source_split_name', 'train',
                    'Name of the train split for the source.')

flags.DEFINE_string('target_split_name', 'train',
                    'Name of the train split for the target.')

flags.DEFINE_string('dataset_dir', './dataset',
                    'The directory where the datasets can be found.')

flags.DEFINE_integer('num_preprocessing_threads', 1,
                     'The number of threads used to create the batches.')

flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')

def main(_):
    util.config_logging()

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    target_dataset = dataset_factory.get_dataset(
        FLAGS.target_dataset,
        split_name='test',
        dataset_dir=FLAGS.dataset_dir)
    target_images, target_labels = dataset_factory.provide_batch(
        FLAGS.target_dataset, 'test', FLAGS.dataset_dir, FLAGS.num_readers,
        1, FLAGS.num_preprocessing_threads)
    target_label = tf.argmax(target_labels['classes'], -1)
    del target_labels['classes']
    num_target_classes = target_dataset.num_classes

    ####################
    # Define the model #
    ####################
    net, layers = classifier.LeNet(target_images,
                                   False,
                                   num_target_classes,
                                   reuse_private=False,
                                   private_scope='source_only',
                                   reuse_shared=False,
                                   shared_scope='source_only')
    net = tf.argmax(net, -1)

    sess.run(tf.global_variables_initializer())

    with slim.queues.QueueRunners(sess):
        #######################################evaluate source only####################################
        util.evalutation(sess,
                    net,
                    target_label,
                    num_target_classes,
                    'ADDA/snapshot',
                    'LeNet_mnist',
                    target_dataset.num_samples,
                    target_images)

        #######################################evaluate adda############################################
        util.evalutation(sess,
                     net,
                     target_label,
                     num_target_classes,
                     'ADDA/snapshot',
                     'adda_lenet_mnist_minstm',
                     target_dataset.num_samples,
                     target_images)

if __name__ == '__main__':
    tf.app.run()