import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append("./")
from common import util, classifier
from data_factory import dataset_factory
from PixelDA import pixelda_model

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

flags.DEFINE_string('dataset_dir', '../dataset',
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

    #########################
    # Preprocess the inputs #
    #########################
    source_dataset = dataset_factory.get_dataset(
        FLAGS.source_dataset,
        split_name='train',
        dataset_dir=FLAGS.dataset_dir)
    num_source_classes = source_dataset.num_classes
    source_images, source_labels = dataset_factory.provide_batch(
        FLAGS.source_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
        32, FLAGS.num_preprocessing_threads)
    source_label = tf.argmax(source_labels['classes'], 1)
    del source_labels['classes']

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
    gen, dis, cls = pixelda_model.create_model(target_images, source_images, num_target_classes)
    cls['target_task_logits'] = tf.argmax(cls['target_task_logits'], -1)

    net, layers = classifier.LeNet(target_images,
                                   False,
                                   num_target_classes,
                                   reuse_private=False,
                                   private_scope='target',
                                   reuse_shared=False,
                                   shared_scope='target')
    net = tf.argmax(net, -1)


    with slim.queues.QueueRunners(sess):
        #######################################pixelda transfer####################################
        cls_var = util.collect_vars('classifier')
        util.evalutation(sess,
                         cls['target_task_logits'],
                         target_label,
                         num_target_classes,
                         '../PixelDA/snapshot',
                         'pixelda',
                         target_dataset.num_samples,
                         cls_var,
                         target_images)

        #######################################evaluate adda############################################
        target_vars = util.collect_vars('target')
        target_vars = util.copyKeySet(cls_var, target_vars)
        util.evalutation(sess,
                         net,
                         target_label,
                         num_target_classes,
                         'snapshot',
                         'adda_pixelda',
                         target_dataset.num_samples,
                         target_vars,
                         target_images)

if __name__ == '__main__':
    tf.app.run()
