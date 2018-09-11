import logging
import os
import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
from collections import deque
from tqdm import tqdm
from data_factory import dataset_factory
import Preprocessing
import Classifier
import util

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

flags.DEFINE_integer('snapshot', 5000, '')

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

    lr = FLAGS.lr
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    step = optimizer.minimize(loss)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    model_vars = util.collect_vars('source')
    saver = tf.train.Saver(var_list=model_vars)
    output_dir = os.path.join('snapshot', 'LeNet_mnist')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    losses = deque(maxlen=10)
    bar = tqdm(range(FLAGS.iteration))
    bar.set_description('{} (lr: {:.0e})'.format('LeNet_mnist', lr))
    bar.refresh()

    display = 10
    stepsize = None
    for i in bar:
        loss_val, _ = sess.run([loss, step])
        losses.append(loss_val)
        if i % display == 0:
            logging.info('{:20} {:10.4f}     (avg: {:10.4f})'
                        .format('Iteration {}:'.format(i),
                                loss_val,
                                np.mean(losses)))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format('LeNet_mnist', lr))
        if (i + 1) % FLAGS.snapshot == 0:
            snapshot_path = saver.save(sess, os.path.join(output_dir, 'LeNet_mnist'),
                                       global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))

    coord.request_stop()
    coord.join(threads)
    sess.close()

    return

if __name__ == '__main__':
    tf.app.run()