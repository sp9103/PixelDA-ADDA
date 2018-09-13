import logging
import os
import sys
import random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from collections import deque
from tqdm import tqdm
from data_factory import dataset_factory
from ADDA import Classifier
from ADDA import util
from ADDA import adversary

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

flags.DEFINE_integer('num_preprocessing_threads', 4,
                     'The number of threads used to create the batches.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_integer('iteration', 20000, '')

flags.DEFINE_integer('snapshot', 5000, '')

flags.DEFINE_float('lr', 1e-4, '')

def main(_):
    util.config_logging()

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

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
    num_target_classes = target_dataset.num_classes
    target_images, target_labels = dataset_factory.provide_batch(
        FLAGS.target_dataset, 'test', FLAGS.dataset_dir, FLAGS.num_readers,
        1, FLAGS.num_preprocessing_threads)
    target_label = tf.argmax(target_labels['classes'], -1)
    del target_labels['classes']

    if num_source_classes != num_target_classes:
        raise ValueError(
            'Source and Target datasets must have same number of classes. '
            'Are %d and %d' % (num_source_classes, num_target_classes))

    ####################
    # Define the model #
    ####################
    source_net, source_layers = Classifier.LeNet(source_images,
                                                 False,
                                                 num_source_classes,
                                                 reuse_private=False,
                                                 private_scope='source',
                                                 reuse_shared=False,
                                                 shared_scope='source')
    target_net, target_layers = Classifier.LeNet(target_images,
                                                 False,
                                                 num_target_classes,
                                                 reuse_private=False,
                                                 private_scope='target',
                                                 reuse_shared=False,
                                                 shared_scope='target')

    # adversarial network - 다차원일 때 1차원으로 펴주기 위한 것이기는하나.. 이미 벡터라서 예제에는 의미가 없다.
    source_net = tf.reshape(source_net, [-1, int(source_net.get_shape()[-1])])
    target_net = tf.reshape(target_net, [-1, int(target_net.get_shape()[-1])])
    # 각 ft에서 올라온것을 Batch처럼 보고 사용함
    adversary_ft = tf.concat([source_net, target_net], 0)
    source_adversary_label = tf.zeros([tf.shape(source_net)[0]], tf.int32)         #source가 들어오면 0으로 맞추게
    target_adversary_label = tf.ones([tf.shape(target_net)[0]], tf.int32)          #target이 들어오면 1로 맞추게
    adversary_label = tf.concat(
        [source_adversary_label, target_adversary_label], 0)
    adversary_logits = adversary.adversarial_discriminator(
        adversary_ft, [500, 500])

    #################Loss Define##########################################
    mapping_loss = tf.losses.sparse_softmax_cross_entropy(
        1 - adversary_label, adversary_logits)  # Mapping Loss는 네트워크가 못 맞출 수록 낮음
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(
        adversary_label, adversary_logits)  # Adv loss는 잘 맞출 수록 낮음

    source_vars = util.collect_vars('source')
    target_vars = util.collect_vars('target')
    adversary_vars = util.collect_vars('adversary')

    lr_var = tf.Variable(FLAGS.lr, name='learning_rate', trainable=False)
    optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)

    mapping_step = optimizer.minimize(
        mapping_loss, var_list=list(target_vars.values()))  # adversary_ft가 잘 못맞추게 target var를 학습한다
    adversary_step = optimizer.minimize(
        adversary_loss, var_list=list(adversary_vars.values()))  # adversary_ft가 잘 맞출수 있게 학습한다. 구분되게

    # restore weights
    sess.run(tf.global_variables_initializer())
    output_dir = os.path.join('ADDA/snapshot', 'LeNet_mnist')
    if os.path.isdir(output_dir):
        weights = tf.train.latest_checkpoint(output_dir)
        logging.info('Restoring weights from {}:'.format(weights))
        logging.info('    Restoring source model:')
        for src, tgt in source_vars.items():
            logging.info('        {:30} -> {:30}'.format(src, tgt.name))
        source_restorer = tf.train.Saver(var_list=source_vars)
        source_restorer.restore(sess, weights)
        logging.info('    Restoring target model:')
        for src, tgt in target_vars.items():
            logging.info('        {:30} -> {:30}'.format(src, tgt.name))
        target_restorer = tf.train.Saver(var_list=target_vars)
        target_restorer.restore(sess, weights)
    else:
        return

    output_dir = os.path.join('ADDA/snapshot', 'adda_lenet_svhn_mnist')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mapping_losses = deque(maxlen=10)
    adversary_losses = deque(maxlen=10)
    bar = tqdm(range(FLAGS.iteration))
    bar.set_description('{} (lr: {:.0e})'.format('adda_lenet_svhn_mnist', FLAGS.lr))
    bar.refresh()

    display = 10
    stepsize = None
    with slim.queues.QueueRunners(sess):
        for i in bar:
            #g-step
            mapping_loss_val, _ = sess.run([mapping_loss, mapping_step])
            mapping_losses.append(mapping_loss_val)

            #d-step
            adversary_loss_val, _ = sess.run([adversary_loss, adversary_step])
            adversary_losses.append(adversary_loss_val)
            if i % display == 0:
                logging.info('{:20} Mapping: {:10.4f}     (avg: {:10.4f})'
                             '    Adversary: {:10.4f}     (avg: {:10.4f})'
                             .format('Iteration {}:'.format(i),
                                     mapping_loss_val,
                                     np.mean(mapping_losses),
                                     adversary_loss_val,
                                     np.mean(adversary_losses)))
            if stepsize is not None and (i + 1) % stepsize == 0:
                lr = sess.run(lr_var.assign(FLAGS.lr * 0.1))
                logging.info('Changed learning rate to {:.0e}'.format(lr))
                bar.set_description('{} (lr: {:.0e})'.format('adda_lenet_svhn_mnist', lr))
            if (i + 1) % FLAGS.snapshot == 0:
                snapshot_path = target_restorer.save(
                    sess, os.path.join(output_dir, 'adda_lenet_svhn_mnist'), global_step=i + 1)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

if __name__ == '__main__':
    tf.app.run()