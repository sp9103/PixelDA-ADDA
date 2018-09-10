import os
import sys
sys.path.append('../')
#os.chdir('..') # 작업 위치 변경

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data_factory import dataset_factory
import Preprocessing
import Classifier

flags = tf.app.flags
FLAGS = flags.FLAGS
run_dir = './model'
checkpoint_dir = './model'

flags.DEFINE_string('source_dataset', 'mnist', 'The name of the source dataset.'
                    ' If hparams="arch=dcgan", this flag is ignored.')

flags.DEFINE_string('target_dataset', 'mnist_m',
                    'The name of the target dataset.')

flags.DEFINE_string('source_split_name', 'train',
                    'Name of the train split for the source.')

flags.DEFINE_string('target_split_name', 'train',
                    'Name of the train split for the target.')

flags.DEFINE_string('dataset_dir', '../',
                    'The directory where the datasets can be found.')

flags.DEFINE_integer('num_preprocessing_threads', 4,
                     'The number of threads used to create the batches.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_integer('iteration', 20000, '')

flags.DEFINE_boolean('pre_training', True, '')

flags.DEFINE_float('lr', 1e-4, '')

def main(_):
    for path in [run_dir, checkpoint_dir]:
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)

    #########################
    # Preprocess the inputs #
    #########################
    target_dataset = dataset_factory.get_dataset(
        FLAGS.target_dataset,
        split_name='train',
        dataset_dir=FLAGS.dataset_dir)
    target_images, _ = dataset_factory.provide_batch(
        FLAGS.target_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
        32, FLAGS.num_preprocessing_threads)
    num_target_classes = target_dataset.num_classes
    target_images = Preprocessing.preprocessing(target_images)

    source_dataset = dataset_factory.get_dataset(
        FLAGS.source_dataset,
        split_name='train',
        dataset_dir=FLAGS.dataset_dir)
    num_source_classes = source_dataset.num_classes
    source_images, source_labels = dataset_factory.provide_batch(
        FLAGS.source_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
        32, FLAGS.num_preprocessing_threads)
    source_labels['class'] = tf.argmax(source_labels['classes'], 1)
    del source_labels['classes']
    if num_source_classes != num_target_classes:
        raise ValueError(
            'Source and Target datasets must have same number of classes. '
            'Are %d and %d' % (num_source_classes, num_target_classes))

    ####################
    # Define the model #
    ####################
    net, layers = Classifier.LeNet(source_images,
                                   False,
                                   num_target_classes,
                                   reuse_private=False,
                                   private_scope='source',
                                   reuse_shared=False,
                                   shared_scope='source')
    class_loss = tf.losses.sparse_softmax_cross_entropy(source_labels['class'], net)
    loss = tf.losses.get_total_loss()

    lr_var = tf.Variable(FLAGS.lr, name='learning_rate', trainable=False)
    optimizer = tf.train.AdamOptimizer(lr_var)
    step = optimizer.minimize(loss)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    if weights:


    coord.request_stop()
    coord.join(threads)
    sess.close()

    return

if __name__ == '__main__':
    tf.app.run()